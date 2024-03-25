struct GMGLinearSolverFromMatrices{A,B,C,D,E,F,G} <: Algebra.LinearSolver
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
  mh::ModelHierarchy,
  smatrices::AbstractArray{<:AbstractMatrix},
  interp,restrict;
  pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(mh)-1),
  post_smoothers  = pre_smoothers,
  coarsest_solver = Gridap.Algebra.LUSolver(),
  mode::Symbol    = :preconditioner,
  maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
)
  @check mode ∈ [:preconditioner,:solver]
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog("GMG",tols;verbose=verbose)

  return GMGLinearSolverFromMatrices(
    mh,smatrices,interp,restrict,pre_smoothers,post_smoothers,coarsest_solver,mode,log
  )
end

struct GMGLinearSolverFromWeakform{A,B,C,D,E,F,G,H,I} <: Algebra.LinearSolver
  mh              :: A
  biforms         :: B
  trials          :: C
  tests           :: D
  interp          :: E
  restrict        :: F
  pre_smoothers   :: G
  post_smoothers  :: H
  coarsest_solver :: I
  mode            :: Symbol
  log             :: ConvergenceLog{Float64}
  is_nonlinear    :: Bool
end

function GMGLinearSolver(
  mh::ModelHierarchy,
  trials::FESpaceHierarchy,
  tests::FESpaceHierarchy,
  biforms::AbstractArray{<:Function},
  interp,restrict;
  pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(mh)-1),
  post_smoothers  = pre_smoothers,
  coarsest_solver = Gridap.Algebra.LUSolver(),
  mode::Symbol    = :preconditioner,
  is_nonlinear    = false,
  maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
)
  @check mode ∈ [:preconditioner,:solver]
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog("GMG",tols;verbose=verbose)

  return GMGLinearSolverFromWeakform(
    mh,trials,tests,biforms,interp,restrict,pre_smoothers,post_smoothers,coarsest_solver,mode,log,is_nonlinear
  )
end

struct GMGSymbolicSetup{A} <: Algebra.SymbolicSetup
  solver :: A
end

function Algebra.symbolic_setup(solver::GMGLinearSolverFromMatrices,::AbstractMatrix)
  return GMGSymbolicSetup(solver)
end

function Algebra.symbolic_setup(solver::GMGLinearSolverFromWeakform,::AbstractMatrix)
  return GMGSymbolicSetup(solver)
end

struct GMGNumericalSetup{A,B,C,D,E,F} <: Algebra.NumericalSetup
  solver                 :: A
  finest_level_cache     :: B
  pre_smoothers_caches   :: C
  post_smoothers_caches  :: D
  coarsest_solver_cache  :: E
  work_vectors           :: F
end

function Algebra.numerical_setup(ss::GMGSymbolicSetup{<:GMGLinearSolverFromMatrices},mat::AbstractMatrix)
  s = ss.solver
  smatrices = gmg_compute_smatrices(s,mat)

  finest_level_cache = gmg_finest_level_cache(smatrices)
  work_vectors = gmg_work_vectors(smatrices)
  pre_smoothers_caches = gmg_smoothers_caches(s.pre_smoothers,smatrices)
  if !(s.pre_smoothers === s.post_smoothers)
    post_smoothers_caches = gmg_smoothers_caches(s.post_smoothers,smatrices)
  else
    post_smoothers_caches = pre_smoothers_caches
  end
  coarsest_solver_cache = gmg_coarse_solver_caches(s.coarsest_solver,smatrices,work_vectors)

  return GMGNumericalSetup(
    s,finest_level_cache,pre_smoothers_caches,post_smoothers_caches,coarsest_solver_cache,work_vectors
  )
end

function Gridap.Algebra.numerical_setup!(
  ns::GMGNumericalSetup{<:GMGLinearSolverFromMatrices},
  mat::AbstractMatrix
)
  msg = "
    GMGLinearSolverFromMatrices does not support updates.\n
    Please use GMGLinearSolverFromWeakform instead.
  "
  @error msg
end

function Gridap.Algebra.numerical_setup!(
  ns::GMGNumericalSetup{<:GMGLinearSolverFromWeakform},
  mat::AbstractMatrix,
  x::AbstractVector
)
  s = ns.solver
  mh = s.mh
  map(linear_indices(mh)) do l
    if l == 1
      copyto!(ns.smatrices[l],mat)
    end
    Ul = get_fe_space(s.trials,l)
    Vl = get_fe_space(s.tests,l)
    A  = ns.smatrices[l]
    uh = FEFunction(Ul,x)
    al(u,v) = s.is_nonlinear ? s.biforms[l](uh,u,v) : s.biforms[l](u,v)
    return assemble_matrix!(al,A,Ul,Vl)
  end
end

function gmg_compute_smatrices(s::GMGLinearSolverFromMatrices,mat::AbstractMatrix)
  smatrices = s.smatrices
  smatrices[1] = mat
  return smatrices
end

function gmg_compute_smatrices(s::GMGLinearSolverFromWeakform,mat::AbstractMatrix)
  mh = s.mh
  map(linear_indices(mh)) do l
    if l == 1
      return mat
    end
    Ul = get_fe_space(s.trials,l)
    Vl = get_fe_space(s.tests,l)
    uh = zero(Ul)
    al(u,v) = s.is_nonlinear ? s.biforms[l](uh,u,v) : s.biforms[l](u,v)
    return assemble_matrix(al,Ul,Vl)
  end
end

function gmg_finest_level_cache(smatrices::AbstractVector{<:AbstractMatrix})
  with_level(smatrices,1) do Ah
    rh = allocate_in_domain(Ah); fill!(rh,0.0)
    return rh
  end
end

function gmg_smoothers_caches(smoothers::AbstractVector{<:LinearSolver},smatrices::AbstractVector{<:AbstractMatrix})
  nlevs = num_levels(smatrices)
  # Last (i.e., coarsest) level does not need pre-/post-smoothing
  caches = map(smoothers,view(smatrices,1:nlevs-1)) do smoother, mat
    numerical_setup(symbolic_setup(smoother, mat), mat)
  end
  return caches
end

function gmg_coarse_solver_caches(solver::LinearSolver,smatrices,work_vectors)
  nlevs = num_levels(smatrices)
  with_level(smatrices,nlevs) do AH
    _, _, xH, rH = work_vectors[nlevs-1]
    cache = numerical_setup(symbolic_setup(solver, AH), AH)
    if isa(solver,PETScLinearSolver)
      cache = CachedPETScNS(cache, xH, rH)
    end
    return cache
  end
end

function gmg_work_vectors(smatrices::AbstractVector{<:AbstractMatrix})
  nlevs = num_levels(smatrices)
  mats = view(smatrices,1:nlevs-1)
  work_vectors = map(linear_indices(mats),mats) do lev, Ah
    dxh  = allocate_in_domain(Ah); fill!(dxh,zero(eltype(dxh)))
    Adxh = allocate_in_range(Ah); fill!(Adxh,zero(eltype(Adxh)))

    rH, dxH = with_level(smatrices,lev+1;default=(nothing,nothing)) do AH
      rH  = allocate_in_domain(AH); fill!(rH,zero(eltype(rH)))
      dxH = allocate_in_domain(AH); fill!(dxH,zero(eltype(dxH)))
      rH, dxH
    end
    dxh, Adxh, dxH, rH
  end
  return work_vectors
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
