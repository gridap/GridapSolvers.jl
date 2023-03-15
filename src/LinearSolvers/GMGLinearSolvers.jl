struct GMGLinearSolver{A,B,C,D,E,F,G,H} <: Gridap.Algebra.LinearSolver
  mh              :: ModelHierarchy
  smatrices       :: A
  interp          :: B
  restrict        :: C
  pre_smoothers   :: D
  post_smoothers  :: E
  coarsest_solver :: F
  maxiter         :: G
  rtol            :: H
  verbose         :: Bool
  mode            :: Symbol
end

function GMGLinearSolver(mh,smatrices,interp,restrict;
      pre_smoothers   = Fill(RichardsonSmoother(JacobiLinearSolver(),10),num_levels(mh)-1),
      post_smoothers  = pre_smoothers,
      coarsest_solver = Gridap.Algebra.BackslashSolver(),
      maxiter         = 100,
      rtol            = 1.0e-06,
      verbose::Bool   = false,
      mode            = :preconditioner)

  Gridap.Helpers.@check mode ∈ [:preconditioner,:solver]
  Gridap.Helpers.@check isa(maxiter,Integer)
  Gridap.Helpers.@check isa(rtol,Real)

  A=typeof(smatrices)
  B=typeof(interp)
  C=typeof(restrict)
  D=typeof(pre_smoothers)
  E=typeof(post_smoothers)
  F=typeof(coarsest_solver)
  G=typeof(maxiter)
  H=typeof(rtol)
  return GMGLinearSolver{A,B,C,D,E,F,G,H}(mh,smatrices,interp,restrict,pre_smoothers,post_smoothers,
                                          coarsest_solver,maxiter,rtol,verbose,mode)
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

  function GMGNumericalSetup(ss::GMGSymbolicSetup)
    mh              = ss.solver.mh
    pre_smoothers   = ss.solver.pre_smoothers
    post_smoothers  = ss.solver.post_smoothers
    smatrices       = ss.solver.smatrices
    coarsest_solver = ss.solver.coarsest_solver

    finest_level_cache = setup_finest_level_cache(mh,smatrices)
    work_vectors = allocate_work_vectors(mh,smatrices)
    pre_smoothers_caches = setup_smoothers_caches(mh,pre_smoothers,smatrices)
    if (!(pre_smoothers === post_smoothers))
      post_smoothers_caches = setup_smoothers_caches(mh,post_smoothers,smatrices)
    else
      post_smoothers_caches = pre_smoothers_caches
    end
    coarsest_solver_cache = setup_coarsest_solver_cache(mh,coarsest_solver,smatrices)

    A = typeof(finest_level_cache)
    B = typeof(pre_smoothers_caches)
    C = typeof(post_smoothers_caches)
    D = typeof(coarsest_solver_cache)
    E = typeof(work_vectors)
    return new{A,B,C,D,E}(ss.solver,finest_level_cache,pre_smoothers_caches,post_smoothers_caches,coarsest_solver_cache,work_vectors)
  end
end

function Gridap.Algebra.numerical_setup(ss::GMGSymbolicSetup,mat::AbstractMatrix)
  return GMGNumericalSetup(ss)
end

function setup_finest_level_cache(mh::ModelHierarchy,smatrices::Vector{<:AbstractMatrix})
  cache = nothing
  parts = get_level_parts(mh,1)
  if i_am_in(parts)
    Ah = smatrices[1]
    rh = PVector(0.0, Ah.cols)
    cache = rh
  end
  return cache
end

function setup_smoothers_caches(mh::ModelHierarchy,smoothers::AbstractVector{<:LinearSolver},smatrices::Vector{<:AbstractMatrix})
  Gridap.Helpers.@check length(smoothers) == num_levels(mh)-1
  nlevs = num_levels(mh)
  # Last (i.e., coarsest) level does not need pre-/post-smoothing
  caches = Vector{Any}(undef,nlevs-1)
  for i = 1:nlevs-1
    parts = get_level_parts(mh,i)
    if i_am_in(parts)
      ss = symbolic_setup(smoothers[i], smatrices[i])
      caches[i] = numerical_setup(ss, smatrices[i])
    end
  end
  return caches
end

function setup_coarsest_solver_cache(mh::ModelHierarchy,coarsest_solver::LinearSolver,smatrices::Vector{<:AbstractMatrix})
  cache = nothing
  nlevs = num_levels(mh)
  parts = get_level_parts(mh,nlevs)
  if i_am_in(parts)
    mat = smatrices[nlevs]
    if (num_parts(parts) == 1) # Serial
      cache = map_parts(mat.owned_owned_values) do Ah
        ss  = symbolic_setup(coarsest_solver, Ah)
        numerical_setup(ss, Ah)
      end
      cache = cache.part
    else # Parallel
      ss = symbolic_setup(coarsest_solver, mat)
      cache = numerical_setup(ss, mat)
    end
  end
  return cache
end

function setup_coarsest_solver_cache(mh::ModelHierarchy,coarsest_solver::PETScLinearSolver,smatrices::Vector{<:AbstractMatrix})
  cache = nothing
  nlevs = num_levels(mh)
  parts = get_level_parts(mh,nlevs)
  if i_am_in(parts)
    mat   = smatrices[nlevs]
    if (num_parts(parts) == 1) # Serial
      cache = map_parts(mat.owned_owned_values) do Ah
        rh  = convert(PETScVector,fill(0.0,size(A,2)))
        xh  = convert(PETScVector,fill(0.0,size(A,2)))
        ss  = symbolic_setup(coarsest_solver, Ah)
        ns  = numerical_setup(ss, Ah)
        return ns, xh, rh
      end
      cache = cache.part
    else # Parallel
      rh = convert(PETScVector,PVector(0.0,mat.cols))
      xh = convert(PETScVector,PVector(0.0,mat.cols))
      ss = symbolic_setup(coarsest_solver, mat)
      ns = numerical_setup(ss, mat)
      cache = ns, xh, rh
    end
  end
  return cache
end

function allocate_level_work_vectors(mh::ModelHierarchy,smatrices::Vector{<:AbstractMatrix},lev::Integer)
  dxh   = PVector(0.0, smatrices[lev].cols)
  Adxh  = PVector(0.0, smatrices[lev].rows)

  cparts = get_level_parts(mh,lev+1)
  if i_am_in(cparts)
    AH  = smatrices[lev+1]
    rH  = PVector(0.0,AH.cols)
    dxH = PVector(0.0,AH.cols)
  else
    rH  = nothing
    dxH = nothing
  end
  return dxh, Adxh, dxH, rH
end

function allocate_work_vectors(mh::ModelHierarchy,smatrices::Vector{<:AbstractMatrix})
  nlevs = num_levels(mh)
  work_vectors = Vector{Any}(undef,nlevs-1)
  for i = 1:nlevs-1
    parts = get_level_parts(mh,i)
    if i_am_in(parts)
      work_vectors[i] = allocate_level_work_vectors(mh,smatrices,i)
    end
  end
  return work_vectors
end

function solve_coarsest_level!(parts::AbstractPData,::LinearSolver,xh::PVector,rh::PVector,caches)
  if (num_parts(parts) == 1)
    map_parts(xh.owned_values,rh.owned_values) do xh, rh
       solve!(xh,caches,rh)
    end
  else
    solve!(xh,caches,rh)
  end
end

function solve_coarsest_level!(parts::AbstractPData,::PETScLinearSolver,xh::PVector,rh::PVector,caches)
  solver_ns, xh_petsc, rh_petsc = caches
  if (num_parts(parts) == 1)
    map_parts(xh.owned_values,rh.owned_values) do xh, rh
      copy!(rh_petsc,rh)
      solve!(xh_petsc,solver_ns,rh_petsc)
      copy!(xh,xh_petsc)
    end
  else
    copy!(rh_petsc,rh)
    solve!(xh_petsc,solver_ns,rh_petsc)
    copy!(xh,xh_petsc)
  end
end

function apply_GMG_level!(lev::Integer,xh::Union{PVector,Nothing},rh::Union{PVector,Nothing},ns::GMGNumericalSetup;verbose=false)
  mh = ns.solver.mh
  parts = get_level_parts(mh,lev)
  if i_am_in(parts)
    if (lev == num_levels(mh)) 
      ## Coarsest level
      coarsest_solver = ns.solver.coarsest_solver
      coarsest_solver_cache = ns.coarsest_solver_cache
      solve_coarsest_level!(parts,coarsest_solver,xh,rh,coarsest_solver_cache)
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
      apply_GMG_level!(lev+1,dxH,rH,ns;verbose=verbose)

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

  mh      = ns.solver.mh
  maxiter = ns.solver.maxiter
  rtol    = ns.solver.rtol
  verbose = ns.solver.verbose
  mode    = ns.solver.mode

  # TODO: When running in preconditioner mode, do we really need to compute the norm? It's a global com....
  rh = ns.finest_level_cache
  if (mode == :preconditioner)
    fill!(x,0.0)
    copy!(rh,b)
  else
    Ah = ns.solver.smatrices[1]
    mul!(rh,Ah,x)
    rh .= b .- rh
  end

  nrm_r0 = norm(rh)
  nrm_r  = nrm_r0
  current_iter = 0
  rel_res = nrm_r / nrm_r0
  parts = get_level_parts(mh,1)

  if i_am_main(parts)
    @printf "%6s  %12s" "Iter" "Rel res\n"
    @printf "%6i  %12.4e\n" current_iter rel_res
  end

  while (current_iter < maxiter) && (rel_res > rtol)
    apply_GMG_level!(1,x,rh,ns;verbose=verbose)

    nrm_r   = norm(rh)
    rel_res = nrm_r / nrm_r0
    current_iter += 1
    if i_am_main(parts)
      @printf "%6i  %12.4e\n" current_iter rel_res
    end
  end
  converged = (rel_res < rtol)
  return current_iter, converged
end

function LinearAlgebra.ldiv!(x::AbstractVector,ns::GMGNumericalSetup,b::AbstractVector)
  solve!(x,ns,b)
end
