struct BlockTriangularSolver{T,N,A,B,C} <: Gridap.Algebra.LinearSolver
  blocks  :: A
  solvers :: B
  coeffs  :: C
  function BlockTriangularSolver(
    blocks  :: AbstractMatrix{<:SolverBlock},
    solvers :: AbstractVector{<:Gridap.Algebra.LinearSolver},
    coeffs = fill(1.0,size(blocks)),
    half   = :upper
    )
    N = length(solvers)
    @check size(blocks,1) == size(blocks,2) == N
    @check size(coeffs,1) == size(coeffs,2) == N
    @check half ∈ (:upper,:lower)

    A = typeof(blocks)
    B = typeof(solvers)
    C = typeof(coeffs)
    return new{Val{half},N,A,B,C}(blocks,solvers,coeffs)
  end
end

function BlockTriangularSolver(solvers::AbstractVector{<:Gridap.Algebra.LinearSolver}; 
                               is_nonlinear::Matrix{Bool}=fill(false,(length(solvers),length(solvers))),
                               coeffs=fill(1.0,size(is_nonlinear)),
                               half=:upper)
  blocks = map(nl -> nl ? NonlinearSystemBlock() : LinearSystemBlock(),is_nonlinear)
  return BlockTriangularSolver(blocks,solvers,coeffs,half)
end

# Symbolic setup

struct BlockTriangularSolverSS{A,B,C} <: Gridap.Algebra.SymbolicSetup
  solver       :: A
  block_ss     :: B
  block_caches :: C
end

function Gridap.Algebra.symbolic_setup(solver::BlockTriangularSolver,mat::AbstractBlockMatrix)
  mat_blocks   = blocks(mat)
  block_caches = map(instantiate_block_cache,solver.blocks,mat_blocks)
  block_ss     = map(symbolic_setup,solver.solvers,diag(block_caches))
  return BlockTriangularSolverSS(solver,block_ss,block_caches)
end

function Gridap.Algebra.symbolic_setup(solver::BlockTriangularSolver{T,N},mat::AbstractBlockMatrix,x::AbstractBlockVector) where {T,N}
  mat_blocks   = blocks(mat)
  vec_blocks   = blocks(x)
  block_caches = map(CartesianIndices(solver.blocks)) do I
    instantiate_block_cache(solver.blocks[I],mat_blocks[I],vec_blocks[I[2]])
  end
  block_ss     = map(symbolic_setup,solver.solvers,diag(block_caches),vec_blocks)
  return BlockTriangularSolverSS(solver,block_ss,block_caches)
end

# Numerical setup

struct BlockTriangularSolverNS{T,A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver       :: A
  block_ns     :: B
  block_caches :: C
  work_caches  :: D
  function BlockTriangularSolverNS(
    solver::BlockTriangularSolver{T},
    block_ns,block_caches,work_caches
  ) where T
    A = typeof(solver) 
    B = typeof(block_ns)
    C = typeof(block_caches)
    D = typeof(work_caches)
    return new{T,A,B,C,D}(solver,block_ns,block_caches,work_caches)
  end
end

function Gridap.Algebra.numerical_setup(ss::BlockTriangularSolverSS,mat::AbstractBlockMatrix)
  solver      = ss.solver
  block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches))
  work_caches = allocate_in_range(mat)
  return BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
end

function Gridap.Algebra.numerical_setup(ss::BlockTriangularSolverSS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver      = ss.solver
  vec_blocks  = blocks(x)
  block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches),vec_blocks)
  work_caches = allocate_in_range(mat)
  return BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
end

function Gridap.Algebra.numerical_setup!(ns::BlockTriangularSolverNS,mat::AbstractBlockMatrix)
  solver       = ns.solver
  mat_blocks   = blocks(mat)
  block_caches = map(update_block_cache!,ns.block_caches,solver.blocks,mat_blocks)
  map(numerical_setup!,ns.block_ns,diag(block_caches))
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::BlockTriangularSolverNS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver       = ns.solver
  mat_blocks   = blocks(mat)
  vec_blocks   = blocks(x)
  block_caches = map(CartesianIndices(solver.blocks)) do I
    update_block_cache!(ns.block_caches[I],mat_blocks[I],vec_blocks[I[2]])
  end
  map(numerical_setup!,ns.block_ns,diag(block_caches),vec_blocks)
  return ns
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockTriangularSolverNS{Val{:lower}},b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c, w = ns.solver.coeffs, ns.work_caches
  mats = ns.block_caches
  for iB in 1:NB
    # Add lower off-diagonal contributions
    wi  = w[Block(iB)]
    copy!(wi,b[Block(iB)])
    for jB in 1:iB-1
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = x[Block(jB)]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB]
    xi  = x[Block(iB)]
    solve!(xi,nsi,wi)
  end
  return x
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockTriangularSolverNS{Val{:upper}},b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c, w = ns.solver.coeffs, ns.work_caches
  mats = ns.block_caches
  for iB in NB:-1:1
    # Add upper off-diagonal contributions
    wi  = w[Block(iB)]
    copy!(wi,b[Block(iB)])
    for jB in iB+1:NB
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = x[Block(jB)]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB]
    xi = x[Block(iB)]
    solve!(xi,nsi,wi)
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockTriangularSolverNS,b)
  solve!(x,ns,b)
end
