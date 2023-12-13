
struct BlockDiagonalSolver{N,A,B} <: Gridap.Algebra.LinearSolver
  blocks  :: B
  solvers :: C
  function BlockDiagonalSolver(
    blocks  :: AbstractVector{<:SolverBlock},
    solvers :: AbstractVector{<:Gridap.Algebra.LinearSolver}
    )
    N = length(solvers)
    @check length(blocks) == N

    A = typeof(blocks)
    B = typeof(solvers)
    return new{N,A,B}(blocks,solvers)
  end
end

# Constructors

function BlockDiagonalSolver(solvers::AbstractVector{<:Gridap.Algebra.LinearSolver}; 
                             is_nonlinear::Vector{Bool}=fill(false,length(solvers)))
  blocks = map(nl -> nl ? NonlinearSystemBlock() : LinearSystemBlock(),is_nonlinear)
  return BlockDiagonalSolver(blocks,solvers)
end

function BlockDiagonalSolver(funcs   :: AbstractArray{<:Function},
                             trials  :: AbstractArray{<:FESpace},
                             tests   :: AbstractArray{<:FESpace},
                             solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver};
                             is_nonlinear::Vector{Bool}=fill(false,length(solvers)))
  blocks = map(funcs,trials,tests,is_nonlinear) do f,trial,test,nl
    nl ? TriformBlock(f,trial,test) : BiformBlock(f,trial,test)
  end
  return BlockDiagonalSolver(blocks,solvers)
end

function BlockDiagonalSolver(mats::AbstractVector{<:AbstractMatrix},
                             solvers::AbstractVector{<:Gridap.Algebra.LinearSolver})
  blocks = map(MatrixBlock,mats)
  return BlockDiagonalSolver(blocks,solvers)
end

# Symbolic setup

struct BlockDiagonalSolverSS{A,B,C} <: Gridap.Algebra.SymbolicSetup
  solver       :: A
  block_ss     :: B
  block_caches :: C
end

function Gridap.Algebra.symbolic_setup(solver::BlockDiagonalSolver,mat::AbstractBlockMatrix)
  mat_blocks   = diag(blocks(mat))
  block_caches = map(instantiate_block_cache,solver.blocks,mat_blocks)
  block_ss     = map(symbolic_setup,solver.solvers,block_caches)
  return BlockDiagonalSolverSS(solver,block_ss,block_caches)
end

function Gridap.Algebra.symbolic_setup(solver::BlockDiagonalSolver,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  mat_blocks   = diag(blocks(mat))
  vec_blocks   = blocks(x)
  block_caches = map(instantiate_block_cache,solver.blocks,mat_blocks,vec_blocks)
  block_ss     = map(symbolic_setup,solver.solvers,block_caches,vec_blocks)
  return BlockDiagonalSolverSS(solver,block_ss,block_caches)
end

# Numerical setup

struct BlockDiagonalSolverNS{A,B,C} <: Gridap.Algebra.NumericalSetup
  solver       :: A
  block_ns     :: B
  block_caches :: C
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalSolverSS,mat::AbstractBlockMatrix)
  solver     = ss.solver
  block_ns   = map(numerical_setup,ss.block_ss,ss.block_caches)
  return BlockDiagonalSolverNS(solver,block_ns,ss.block_caches)
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalSolverSS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver     = ss.solver
  vec_blocks = blocks(x)
  block_ns   = map(numerical_setup,ss.block_ss,ss.block_caches,vec_blocks)
  return BlockDiagonalSolverNS(solver,block_ns,ss.block_caches)
end

function Gridap.Algebra.numerical_setup!(ns::BlockDiagonalSolverNS,mat::AbstractBlockMatrix)
  solver       = ns.solver
  mat_blocks   = diag(blocks(mat))
  block_caches = map(update_block_cache!,ns.block_caches,solver.blocks,mat_blocks)
  map(numerical_setup!,ns.block_ns,block_caches)
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::BlockDiagonalSolverNS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver       = ns.solver
  mat_blocks   = diag(blocks(mat))
  vec_blocks   = blocks(x)
  block_caches = map(update_block_cache!,ns.block_caches,solver.blocks,mat_blocks,vec_blocks)
  map(numerical_setup!,ns.block_ns,block_caches,vec_blocks)
  return ns
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockDiagonalSolverNS,b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  for (iB,bns) in enumerate(ns.block_ns)
    xi = x[Block(iB)]
    bi = b[Block(iB)]
    solve!(xi,bns,bi)
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockDiagonalSolverNS,b)
  solve!(x,ns,b)
end
