
"""
    struct CachedPETScNS <: NumericalSetup
      ...
    end

  Wrapper around a PETSc NumericalSetup, providing highly efficiend reusable caches:

  When converting julia vectors/PVectors to PETSc vectors, we purposely create aliasing 
  of the vector values. This means we can avoid copying data from one to another before solving, 
  but we need to be careful about it. 
  
  This structure takes care of this, and makes sure you do not attempt to solve the system 
  with julia vectors that are not the ones you used to create the solver cache.
"""
struct CachedPETScNS{TM,A}
  ns     :: GridapPETSc.PETScLinearSolverNS{TM}
  X      :: PETScVector
  B      :: PETScVector
  owners :: A

  @doc """
      function CachedPETScNS(ns::PETScLinearSolverNS,x::AbstractVector,b::AbstractVector)

  Create a new instance of [`CachedPETScNS`](@ref) from its underlying properties.
  Once this structure is created, you can **only** solve the system with the same vectors 
  you used to create it.
  """
  function CachedPETScNS(ns::GridapPETSc.PETScLinearSolverNS{TM},x::AbstractVector,b::AbstractVector) where TM
    X = convert(PETScVector,x)
    B = convert(PETScVector,b)
    owners = (x,b)

    A = typeof(owners)
    new{TM,A}(ns,X,B,owners)
  end
end

function Algebra.solve!(x::AbstractVector,ns::CachedPETScNS,b::AbstractVector)
  @assert x === ns.owners[1]
  @assert b === ns.owners[2]
  solve!(ns.X,ns.ns,ns.B)
  consistent!(x)
  return x
end

function Algebra.numerical_setup!(ns::CachedPETScNS,mat::AbstractMatrix)
  numerical_setup!(ns.ns,mat)
end

function Algebra.numerical_setup!(ns::CachedPETScNS,mat::AbstractMatrix,x::AbstractVector)
  numerical_setup!(ns.ns,mat,x)
end

############################################################################################
# Optimisations for GridapSolvers + PETSc

function LinearSolvers.gmg_coarse_solver_caches(
  solver::PETScLinearSolver,
  smatrices::AbstractVector{<:AbstractMatrix},
  work_vectors
)
  nlevs = num_levels(smatrices)
  with_level(smatrices,nlevs) do AH
    _, _, dxH, rH = work_vectors[nlevs-1]
    cache = CachedPETScNS(
      numerical_setup(symbolic_setup(solver, AH), AH), dxH, rH
    )
    return cache
  end
end

function LinearSolvers.gmg_coarse_solver_caches(
  solver::PETScLinearSolver,
  smatrices::AbstractVector{<:AbstractMatrix},
  svectors::AbstractVector{<:AbstractVector},
  work_vectors
)
  nlevs = num_levels(smatrices)
  with_level(smatrices,nlevs) do AH
    _, _, dxH, rH = work_vectors[nlevs-1]
    xH = svectors[nlevs]
    cache = CachedPETScNS(
      numerical_setup(symbolic_setup(solver, AH, xH), AH, xH), dxH,rH
    )
    return cache
  end
end
