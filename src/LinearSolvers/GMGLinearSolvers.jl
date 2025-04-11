"""
    struct GMGLinearSolverFromMatrices <: LinearSolver
      ...
    end

Geometric MultiGrid solver, from algebraic parts.
"""
struct GMGLinearSolverFromMatrices{A,B,C,D,E,F} <: Algebra.LinearSolver
  smatrices       :: A
  interp          :: B
  restrict        :: C
  pre_smoothers   :: D
  post_smoothers  :: E
  coarsest_solver :: F
  mode            :: Symbol
  log             :: ConvergenceLog{Float64}
end

MultilevelTools.num_levels(s::GMGLinearSolverFromMatrices) = length(s.smatrices)

@doc """
    GMGLinearSolver(
      matrices::AbstractArray{<:AbstractMatrix},
      prolongations,
      restrictions;
      pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(mh)-1),
      post_smoothers  = pre_smoothers,
      coarsest_solver = LUSolver(),
      mode::Symbol    = :preconditioner,
      maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
    )

Creates an instance of [`GMGLinearSolverFromMatrices`](@ref) from the underlying model 
hierarchy, the system matrices at each level and the transfer operators and smoothers 
at each level except the coarsest.

The solver has two modes of operation, defined by the kwarg `mode`:

- `:solver`: The GMG solver takes a rhs `b` and returns a solution `x`.
- `:preconditioner`: The GMG solver takes a residual `r` and returns a correction `dx`.

"""
function GMGLinearSolver(
  smatrices::AbstractArray{<:AbstractMatrix},
  interp::AbstractArray,
  restrict::AbstractArray;
  pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),length(smatrices)-1),
  post_smoothers  = pre_smoothers,
  coarsest_solver = Gridap.Algebra.LUSolver(),
  mode::Symbol    = :preconditioner,
  maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
)
  @check length(smatrices)-1 == length(interp) == length(restrict) == length(pre_smoothers) == length(post_smoothers)
  @check mode ∈ [:preconditioner,:solver]
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog("GMG",tols;verbose=verbose)

  return GMGLinearSolverFromMatrices(
    smatrices,interp,restrict,pre_smoothers,post_smoothers,coarsest_solver,mode,log
  )
end

"""
    struct GMGLinearSolverFromWeakForm <: LinearSolver
      ...
    end

Geometric MultiGrid solver, from FE parts.
"""
struct GMGLinearSolverFromWeakform{A,B,C,D,E,F,G,H} <: Algebra.LinearSolver
  trials          :: A
  tests           :: B
  biforms         :: C
  interp          :: D
  restrict        :: E
  pre_smoothers   :: F
  post_smoothers  :: G
  coarsest_solver :: H
  mode            :: Symbol
  log             :: ConvergenceLog{Float64}
  is_nonlinear    :: Bool
  primal_restrictions
end

MultilevelTools.num_levels(s::GMGLinearSolverFromWeakform) = length(s.trials)

@doc """
    GMGLinearSolver(
      trials::FESpaceHierarchy,
      tests::FESpaceHierarchy,
      biforms::AbstractArray{<:Function},
      interp,
      restrict;
      pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(mh)-1),
      post_smoothers  = pre_smoothers,
      coarsest_solver = Gridap.Algebra.LUSolver(),
      mode::Symbol    = :preconditioner,
      is_nonlinear    = false,
      maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
    )

Creates an instance of [`GMGLinearSolverFromMatrices`](@ref) from the underlying model 
hierarchy, the trial and test FEspace hierarchies, the weakform lhs at each level 
and the transfer operators and smoothers at each level except the coarsest.

The solver has two modes of operation, defined by the kwarg `mode`:

- `:solver`: The GMG solver takes a rhs `b` and returns a solution `x`.
- `:preconditioner`: The GMG solver takes a residual `r` and returns a correction `dx`.

"""
function GMGLinearSolver(
  trials::FESpaceHierarchy,
  tests::FESpaceHierarchy,
  biforms::AbstractArray{<:Function},
  interp,restrict;
  pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(trials)-1),
  post_smoothers  = pre_smoothers,
  coarsest_solver = Gridap.Algebra.LUSolver(),
  mode::Symbol    = :preconditioner,
  is_nonlinear    = false,
  maxiter = 100, atol = 1.0e-14, rtol = 1.0e-08, verbose = false,
)
  @check mode ∈ [:preconditioner,:solver]
  @check matching_level_parts(trials,tests,biforms)
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog("GMG",tols;verbose=verbose)

  if is_nonlinear 
    primal_restrictions = setup_restriction_operators(trials,8;mode=:solution,solver=CGSolver(JacobiLinearSolver()))
  else
    primal_restrictions = nothing
  end
  return GMGLinearSolverFromWeakform(
    trials,tests,biforms,interp,restrict,pre_smoothers,post_smoothers,coarsest_solver,mode,log,is_nonlinear,primal_restrictions
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

struct GMGNumericalSetup{A,B,C,D,E,F,G,H} <: Algebra.NumericalSetup
  solver                 :: A
  smatrices              :: B
  finest_level_cache     :: C
  pre_smoothers_caches   :: D
  post_smoothers_caches  :: E
  coarsest_solver_cache  :: F
  work_vectors           :: G
  system_caches          :: H
end

function Algebra.numerical_setup(ss::GMGSymbolicSetup,mat::AbstractMatrix)
  s = ss.solver
  smatrices = gmg_compute_matrices(s,mat)

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
    s,smatrices,finest_level_cache,pre_smoothers_caches,post_smoothers_caches,coarsest_solver_cache,work_vectors,nothing
  )
end

function Algebra.numerical_setup(ss::GMGSymbolicSetup,mat::AbstractMatrix,x::AbstractVector)
  s = ss.solver
  smatrices, svectors = gmg_compute_matrices(s,mat,x)
  system_caches = (smatrices,svectors)

  finest_level_cache = gmg_finest_level_cache(smatrices)
  work_vectors = gmg_work_vectors(smatrices)
  pre_smoothers_caches = gmg_smoothers_caches(s.pre_smoothers,smatrices,svectors)
  if !(s.pre_smoothers === s.post_smoothers)
    post_smoothers_caches = gmg_smoothers_caches(s.post_smoothers,smatrices,svectors)
  else
    post_smoothers_caches = pre_smoothers_caches
  end
  coarsest_solver_cache = gmg_coarse_solver_caches(s.coarsest_solver,smatrices,svectors,work_vectors)

  # Update transfer operators
  interp, restrict = s.interp, s.restrict
  nlevs = num_levels(s)
  map(linear_indices(smatrices),smatrices,svectors) do lev, Ah, xh
    if lev != nlevs
      if isa(interp[lev],PatchProlongationOperator) || isa(interp[lev],MultiFieldTransferOperator)
        MultilevelTools.update_transfer_operator!(interp[lev],xh)
      end
      if isa(restrict[lev],PatchRestrictionOperator) || isa(restrict[lev],MultiFieldTransferOperator)
        MultilevelTools.update_transfer_operator!(restrict[lev],xh)
      end
    end
  end

  return GMGNumericalSetup(
    s,smatrices,finest_level_cache,pre_smoothers_caches,post_smoothers_caches,coarsest_solver_cache,work_vectors,system_caches
  )
end

function Algebra.numerical_setup!(
  ns::GMGNumericalSetup{<:GMGLinearSolverFromMatrices},
  mat::AbstractMatrix
)
  msg = "
    GMGLinearSolverFromMatrices does not support updates.\n
    Please use GMGLinearSolverFromWeakform instead.
  "
  @error msg
end

function Algebra.numerical_setup!(
  ns::GMGNumericalSetup{<:GMGLinearSolverFromWeakform},
  mat::AbstractMatrix,
  x::AbstractVector
)
  @check ns.solver.is_nonlinear

  s = ns.solver
  interp, restrict = s.interp, s.restrict
  pre_smoothers_ns, post_smoothers_ns = ns.pre_smoothers_caches, ns.post_smoothers_caches
  coarsest_solver_ns = ns.coarsest_solver_cache
  nlevs = num_levels(s)

  # Update smatrices and svectors
  smatrices, svectors = gmg_compute_matrices!(ns.system_caches, s,mat,x)
  
  # Update prolongations and smoothers
  map(linear_indices(smatrices),smatrices,svectors) do lev, Ah, xh
    if lev != nlevs
      if isa(interp[lev],PatchProlongationOperator) || isa(interp[lev],MultiFieldTransferOperator)
        MultilevelTools.update_transfer_operator!(interp[lev],xh)
      end
      if isa(restrict[lev],PatchRestrictionOperator) || isa(restrict[lev],MultiFieldTransferOperator)
        MultilevelTools.update_transfer_operator!(restrict[lev],xh)
      end
      numerical_setup!(pre_smoothers_ns[lev],Ah,xh)
      if !(s.pre_smoothers === s.post_smoothers)
        numerical_setup!(post_smoothers_ns[lev],Ah,xh)
      end
    end
    if lev == nlevs
      numerical_setup!(coarsest_solver_ns,Ah,xh)
    end
  end
end

function gmg_project_solutions(solver::GMGLinearSolverFromWeakform,x::AbstractVector)
  tests = solver.tests
  svectors = map(tests) do shlev
    Vh = MultilevelTools.get_fe_space(shlev)
    return zero_free_values(Vh)
  end
  return gmg_project_solutions!(svectors,solver,x)
end

function gmg_project_solutions!(
  svectors::AbstractVector{<:AbstractVector},
  solver::GMGLinearSolverFromWeakform,
  x::AbstractVector
)
  restrictions = solver.primal_restrictions
  copy!(svectors[1],x)
  map(linear_indices(restrictions),restrictions) do lev, R
    mul!(unsafe_getindex(svectors,lev+1),R,svectors[lev])
  end
  return svectors
end

function gmg_compute_matrices(s::GMGLinearSolverFromMatrices,mat::AbstractMatrix)
  smatrices = s.smatrices
  smatrices[1] = mat
  return smatrices
end

function gmg_compute_matrices(s::GMGLinearSolverFromWeakform,mat::AbstractMatrix)
  @check !s.is_nonlinear
  map(linear_indices(s.trials),s.biforms) do l, biform
    if l == 1
      return mat
    end
    Ul = MultilevelTools.get_fe_space(s.trials,l)
    Vl = MultilevelTools.get_fe_space(s.tests,l)
    al(u,v) = biform(u,v)
    return assemble_matrix(al,Ul,Vl)
  end
end

function gmg_compute_matrices(s::GMGLinearSolverFromWeakform,mat::AbstractMatrix,x::AbstractVector)
  @check s.is_nonlinear
  svectors = gmg_project_solutions(s,x)
  smatrices = map(linear_indices(s.trials),s.biforms,svectors) do l, biform, xl
    if l == 1
      return mat
    end
    Ul = MultilevelTools.get_fe_space(s.trials,l)
    Vl = MultilevelTools.get_fe_space(s.tests,l)
    ul = FEFunction(Ul,xl)
    al(u,v) = biform(ul,u,v)
    return assemble_matrix(al,Ul,Vl)
  end
  return smatrices, svectors
end

function gmg_compute_matrices!(caches,s::GMGLinearSolverFromWeakform,mat::AbstractMatrix,x::AbstractVector)
  @check s.is_nonlinear
  tests, trials = s.tests, s.trials
  smatrices, svectors = caches

  svectors = gmg_project_solutions!(svectors,s,x)
  map(linear_indices(s.trials),s.biforms,smatrices,svectors) do l, biform, matl, xl
    if l == 1
      copyto!(matl,mat)
    else
      Ul = MultilevelTools.get_fe_space(trials,l)
      Vl = MultilevelTools.get_fe_space(tests,l)
      ul = FEFunction(Ul,xl)
      al(u,v) = biform(ul,u,v)
      assemble_matrix!(al,matl,Ul,Vl)
    end
  end
  return smatrices, svectors
end

function gmg_finest_level_cache(smatrices::AbstractVector{<:AbstractMatrix})
  with_level(smatrices,1) do Ah
    rh = allocate_in_domain(Ah); fill!(rh,0.0)
    return rh
  end
end

function gmg_smoothers_caches(
  smoothers::AbstractVector{<:LinearSolver},
  smatrices::AbstractVector{<:AbstractMatrix}
)
  nlevs = num_levels(smatrices)
  # Last (i.e., coarsest) level does not need pre-/post-smoothing
  caches = map(smoothers,view(smatrices,1:nlevs-1)) do smoother, mat
    numerical_setup(symbolic_setup(smoother, mat), mat)
  end
  return caches
end

function gmg_smoothers_caches(
  smoothers::AbstractVector{<:LinearSolver},
  smatrices::AbstractVector{<:AbstractMatrix},
  svectors ::AbstractVector{<:AbstractVector}
)
  nlevs = num_levels(smatrices)
  # Last (i.e., coarsest) level does not need pre-/post-smoothing
  caches = map(smoothers,view(smatrices,1:nlevs-1),view(svectors,1:nlevs-1)) do smoother, mat, x
    numerical_setup(symbolic_setup(smoother, mat, x), mat, x)
  end
  return caches
end

function gmg_coarse_solver_caches(
  solver::LinearSolver,
  smatrices::AbstractVector{<:AbstractMatrix},
  work_vectors
)
  nlevs = num_levels(smatrices)
  with_level(smatrices,nlevs) do AH
    _, _, dxH, rH = work_vectors[nlevs-1]
    cache = numerical_setup(symbolic_setup(solver, AH), AH)
    return cache
  end
end

function gmg_coarse_solver_caches(
  solver::LinearSolver,
  smatrices::AbstractVector{<:AbstractMatrix},
  svectors::AbstractVector{<:AbstractVector},
  work_vectors
)
  nlevs = num_levels(smatrices)
  with_level(smatrices,nlevs) do AH
    _, _, dxH, rH = work_vectors[nlevs-1]
    xH = svectors[nlevs]
    cache = numerical_setup(symbolic_setup(solver, AH, xH), AH, xH)
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

function apply_GMG_level!(
  lev::Integer,xh::Union{<:AbstractVector,Nothing},rh::Union{<:AbstractVector,Nothing},ns::GMGNumericalSetup
)
  with_level(ns.smatrices,lev) do Ah
    if (lev == num_levels(ns.solver)) 
      ## Coarsest level
      solve!(xh, ns.coarsest_solver_cache, rh)
    else
      ## General case
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
