struct GMGLinearSolver{A,B,C,D,E,F,G} <: Gridap.Algebra.LinearSolver
  mh              :: A
  smatrices       :: B
  interp          :: C
  restrict        :: D
  pre_smoothers   :: E
  post_smoothers  :: F
  coarsest_solver :: G
  mode            :: Symbol
  log             :: ConvergenceLog{Float64}
end

function GMGLinearSolver(
  mh,smatrices,interp,restrict;
  pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(mh)-1),
  post_smoothers  = pre_smoothers,
  coarsest_solver = Gridap.Algebra.LUSolver(),
  mode::Symbol    = :preconditioner,
  maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
)
  @check mode ∈ [:preconditioner,:solver]
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog("GMG",tols;verbose=verbose)
  
  A = typeof(mh)
  B = typeof(smatrices)
  C = typeof(interp)
  D = typeof(restrict)
  E = typeof(pre_smoothers)
  F = typeof(post_smoothers)
  G = typeof(coarsest_solver)
  return GMGLinearSolver{A,B,C,D,E,F,G}(mh,smatrices,interp,restrict,pre_smoothers,post_smoothers,
                                        coarsest_solver,mode,log)
end

struct GMGSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: GMGLinearSolver
end

function Gridap.Algebra.symbolic_setup(solver::GMGLinearSolver,mat::AbstractMatrix)
  return GMGSymbolicSetup(solver)
end

struct GMGNumericalSetup{A,B,C,D,E} <: Gridap.Algebra.NumericalSetup
  solver                 :: GMGLinearSolver
  finest_level_cache     :: A
  pre_smoothers_caches   :: B
  post_smoothers_caches  :: C
  coarsest_solver_cache  :: D
  work_vectors           :: E
end

function Gridap.Algebra.numerical_setup(ss::GMGSymbolicSetup,mat::AbstractMatrix)
  mh              = ss.solver.mh
  pre_smoothers   = ss.solver.pre_smoothers
  post_smoothers  = ss.solver.post_smoothers
  smatrices       = ss.solver.smatrices
  coarsest_solver = ss.solver.coarsest_solver

  smatrices[1] = mat
  finest_level_cache = gmg_finest_level_cache(mh,smatrices)
  work_vectors = gmg_work_vectors(mh,smatrices)
  pre_smoothers_caches = gmg_smoothers_caches(mh,pre_smoothers,smatrices)
  if !(pre_smoothers === post_smoothers)
    post_smoothers_caches = gmg_smoothers_caches(mh,post_smoothers,smatrices)
  else
    post_smoothers_caches = pre_smoothers_caches
  end
  coarsest_solver_cache = gmg_coarse_solver_caches(mh,coarsest_solver,smatrices,work_vectors)

  return GMGNumericalSetup(ss.solver,finest_level_cache,pre_smoothers_caches,post_smoothers_caches,coarsest_solver_cache,work_vectors)
end

function Gridap.Algebra.numerical_setup!(ss::GMGNumericalSetup,mat::AbstractMatrix)
  # TODO: This does not modify all matrices... How should we deal with this?
  ns.solver.smatrices[1] = mat
end

function gmg_finest_level_cache(mh::ModelHierarchy,smatrices::AbstractVector{<:AbstractMatrix})
  cache = nothing
  parts = get_level_parts(mh,1)
  if i_am_in(parts)
    Ah = smatrices[1]
    rh = allocate_in_domain(Ah); fill!(rh,0.0)
    cache = rh
  end
  return cache
end

function gmg_smoothers_caches(mh::ModelHierarchy,smoothers::AbstractVector{<:LinearSolver},smatrices::AbstractVector{<:AbstractMatrix})
  @check length(smoothers) == num_levels(mh)-1
  nlevs = num_levels(mh)
  # Last (i.e., coarsest) level does not need pre-/post-smoothing
  caches = map(smoothers,view(smatrices,1:nlevs-1)) do smoother, mat
    numerical_setup(symbolic_setup(smoother, mat), mat)
  end
  return caches
end

function gmg_coarse_solver_caches(mh,solver,mats,work_vectors)
  cache = nothing
  nlevs = num_levels(mh)
  parts = get_level_parts(mh,nlevs)
  if i_am_in(parts)
    mat = mats[nlevs]
    _, _, xH, rH = work_vectors[nlevs-1]
    cache = numerical_setup(symbolic_setup(solver, mat), mat)
    if isa(solver,PETScLinearSolver)
      cache = CachedPETScNS(cache, xH, rH)
    end
  end
  return cache
end

function gmg_work_vectors(mh::ModelHierarchy,smatrices::AbstractVector{<:AbstractMatrix})
  @check MultilevelTools.matching_level_parts(mh,smatrices)
  nlevs = num_levels(mh)
  work_vectors = map(view(linear_indices(mh),1:nlevs-1)) do lev
    gmg_work_vectors(mh,smatrices,lev)
  end
  return work_vectors
end

function gmg_work_vectors(mh::ModelHierarchy,smatrices::AbstractVector{<:AbstractMatrix},lev::Integer)
  Ah   = smatrices[lev]
  dxh  = allocate_in_domain(Ah); fill!(dxh,zero(eltype(dxh)))
  Adxh = allocate_in_range(Ah); fill!(Adxh,zero(eltype(Adxh)))

  cparts = get_level_parts(mh,lev+1)
  if i_am_in(cparts)
    AH  = smatrices[lev+1]
    rH  = allocate_in_domain(AH)
    dxH = allocate_in_domain(AH)
  else
    rH  = nothing
    dxH = nothing
  end
  return dxh, Adxh, dxH, rH
end

function apply_GMG_level!(lev::Integer,xh::Union{PVector,Nothing},rh::Union{PVector,Nothing},ns::GMGNumericalSetup)
  mh = ns.solver.mh
  parts = get_level_parts(mh,lev)
  if i_am_in(parts)
    if (lev == num_levels(mh)) 
      ## Coarsest level
      solve!(xh, ns.coarsest_solver_cache, rh)
    else 
      ## General case
      Ah = ns.solver.smatrices[lev]
      restrict, interp = ns.solver.restrict[lev], ns.solver.interp[lev]
      dxh, Adxh, dxH, rH = ns.work_vectors[lev]

      # Pre-smooth current solution
      solve!(xh, ns.pre_smoothers_caches[lev], rh)

      # Restrict the residual
      mul!(rH,restrict,rh)

      # Apply next_level
      !isa(dxH,Nothing) && fill!(dxH,0.0)
      apply_GMG_level!(lev+1,dxH,rH,ns)

      # Interpolate dxH in finer space
      mul!(dxh,interp,dxH)

      # Update solution & residual
      xh .= xh .+ dxh
      mul!(Adxh, Ah, dxh)
      rh .= rh .- Adxh

      # Post-smooth current solution
      solve!(xh, ns.post_smoothers_caches[lev], rh)
    end
  end
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::GMGNumericalSetup,b::AbstractVector)
  mode = ns.solver.mode
  log  = ns.solver.log

  rh = ns.finest_level_cache
  if (mode == :preconditioner)
    fill!(x,0.0)
    copy!(rh,b)
  else
    Ah = ns.solver.smatrices[1]
    mul!(rh,Ah,x)
    rh .= b .- rh
  end

  res  = norm(rh)
  done = init!(log,res)
  while !done
    apply_GMG_level!(1,x,rh,ns)
    res  = norm(rh)
    done = update!(log,res)
  end

  finalize!(log,res)
  return x
end

function LinearAlgebra.ldiv!(x::AbstractVector,ns::GMGNumericalSetup,b::AbstractVector)
  solve!(x,ns,b)
end
