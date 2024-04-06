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
  trials          :: B
  tests           :: C
  biforms         :: D
  interp          :: E
  restrict        :: F
  pre_smoothers   :: G
  post_smoothers  :: H
  coarsest_solver :: I
  mode            :: Symbol
  log             :: ConvergenceLog{Float64}
  is_nonlinear    :: Bool
  primal_restrictions
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
  @check matching_level_parts(mh,trials,tests,biforms)
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog("GMG",tols;verbose=verbose)

  primal_restrictions = is_nonlinear ? setup_restriction_operators(trials,8;mode=:solution,solver=IS_ConjugateGradientSolver(;reltol=1.e-6)) : nothing
  return GMGLinearSolverFromWeakform(
    mh,trials,tests,biforms,interp,restrict,pre_smoothers,post_smoothers,coarsest_solver,mode,log,is_nonlinear,primal_restrictions
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

struct GMGNumericalSetup{A,B,C,D,E,F,G} <: Algebra.NumericalSetup
  solver                 :: A
  smatrices              :: B
  finest_level_cache     :: C
  pre_smoothers_caches   :: D
  post_smoothers_caches  :: E
  coarsest_solver_cache  :: F
  work_vectors           :: G
end

function Algebra.numerical_setup(ss::GMGSymbolicSetup,mat::AbstractMatrix)
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
    s,smatrices,finest_level_cache,pre_smoothers_caches,post_smoothers_caches,coarsest_solver_cache,work_vectors
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
  @check ns.solver.is_nonlinear

  s = ns.solver
  mh, trials, tests, restrictions = s.mh, s.trials, s.tests, s.primal_restrictions
  work_vectors = ns.work_vectors
  nlevs = num_levels(mh)

  # Project solution to all levels
  xh = Vector{AbstractVector}(undef,nlevs)
  xh[1] = work_vectors[1][1]; copy!(xh[1],x)
  map(view(linear_indices(mh),1:nlevs-1),restrictions) do lev, R
    dxh, Adxh, dxH, rH = work_vectors[lev]
    mul!(rH,R,xh[lev])
    xh[lev+1] = rH
  end

  # Update matrices, prolongations and smoothers
  map(linear_indices(mh),ns.smatrices,xh) do lev, Ah, xh
    if lev == 1
      copyto!(Ah,mat)
    else
      Uh = MultilevelTools.get_fe_space(trials,lev)
      Vh = MultilevelTools.get_fe_space(tests,lev)
      uh = FEFunction(Uh,xh)
      ah(u,v) = s.is_nonlinear ? s.biforms[lev](uh,u,v) : s.biforms[lev](u,v)
      assemble_matrix!(ah,Ah,Uh,Vh)
    end
    if lev != num_levels(mh)
      if isa(s.interp[lev],PatchProlongationOperator) || isa(s.interp[lev],MultiFieldTransferOperator)
        MultilevelTools.update_transfer_operator!(s.interp[lev],xh)
      end
      numerical_setup!(ns.pre_smoothers_caches[lev],Ah,xh)
      if !(s.pre_smoothers === s.post_smoothers)
        numerical_setup!(ns.post_smoothers_caches[lev],Ah,xh)
      end
    end
  end
end

function gmg_compute_smatrices(s::GMGLinearSolverFromMatrices,mat::AbstractMatrix)
  smatrices = s.smatrices
  smatrices[1] = mat
  return smatrices
end

function gmg_compute_smatrices(s::GMGLinearSolverFromWeakform,mat::AbstractMatrix)
  map(linear_indices(s.mh),s.biforms) do l, biform
    if l == 1
      return mat
    end
    Ul = MultilevelTools.get_fe_space(s.trials,l)
    Vl = MultilevelTools.get_fe_space(s.tests,l)
    uh = zero(Ul)
    al(u,v) = s.is_nonlinear ? biform(uh,u,v) : biform(u,v)
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
      Ah = ns.smatrices[lev]
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
    Ah = ns.smatrices[1]
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
