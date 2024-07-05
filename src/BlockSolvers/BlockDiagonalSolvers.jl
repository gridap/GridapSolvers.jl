"""
    struct BlockDiagonalSolver{N} <: LinearSolver

Solver representing a block-diagonal solver, i.e 

```
│ A11   0    0  │   │ x1 │   │ r1 │
│  0   A22   0  │ ⋅ │ x2 │ = │ r2 │
│  0    0   A33 │   │ x3 │   │ r3 │
```

where `N` is the number of diagonal blocks.

# Properties:

- `blocks::AbstractVector{<:SolverBlock}`: Matrix of solver blocks, indicating how 
    each diagonal block of the preconditioner is obtained. 
- `solvers::AbstractVector{<:LinearSolver}`: Vector of solvers, 
    one for each diagonal block.
  
"""
struct BlockDiagonalSolver{N,A,B} <: Gridap.Algebra.LinearSolver
  blocks  :: A
  solvers :: B

  @doc """
      function BlockDiagonalSolver(
        blocks  :: AbstractVector{<:SolverBlock},
        solvers :: AbstractVector{<:LinearSolver}
      )

  Create and instance of [`BlockDiagonalSolver`](@ref) from its underlying properties.
  """
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

@doc """
    function BlockDiagonalSolver(
      solvers::AbstractVector{<:LinearSolver}; 
      is_nonlinear::Vector{Bool}=fill(false,length(solvers))
    )
  
Create and instance of [`BlockDiagonalSolver`](@ref) where all blocks are extracted from 
the linear system.
"""
function BlockDiagonalSolver(solvers::AbstractVector{<:Gridap.Algebra.LinearSolver}; 
                             is_nonlinear::Vector{Bool}=fill(false,length(solvers)))
  blocks = map(nl -> nl ? NonlinearSystemBlock() : LinearSystemBlock(),is_nonlinear)
  return BlockDiagonalSolver(blocks,solvers)
end

@doc """
    function BlockDiagonalSolver(
      funcs   :: AbstractArray{<:Function},
      trials  :: AbstractArray{<:FESpace},
      tests   :: AbstractArray{<:FESpace},
      solvers :: AbstractArray{<:LinearSolver};
      is_nonlinear::Vector{Bool}=fill(false,length(solvers))
    )
  
Create and instance of [`BlockDiagonalSolver`](@ref) where all blocks are given by 
integral forms.
"""
function BlockDiagonalSolver(
  funcs   :: AbstractArray{<:Function},
  trials  :: AbstractArray{<:FESpace},
  tests   :: AbstractArray{<:FESpace},
  solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver};
  is_nonlinear::Vector{Bool}=fill(false,length(solvers))
)
  blocks = map(funcs,trials,tests,is_nonlinear) do f,trial,test,nl
    nl ? TriformBlock(f,trial,test) : BiformBlock(f,trial,test)
  end
  return BlockDiagonalSolver(blocks,solvers)
end

@doc """
    function BlockDiagonalSolver(
      mats::AbstractVector{<:AbstractMatrix},
      solvers::AbstractVector{<:LinearSolver}
    )
  
Create and instance of [`BlockDiagonalSolver`](@ref) where all blocks are given by 
external matrices.
"""
function BlockDiagonalSolver(
  mats::AbstractVector{<:AbstractMatrix},
  solvers::AbstractVector{<:Gridap.Algebra.LinearSolver}
)
  blocks = map(MatrixBlock,mats)
  return BlockDiagonalSolver(blocks,solvers)
end

# Symbolic setup

struct BlockDiagonalSolverSS{A,B} <: Gridap.Algebra.SymbolicSetup
  solver       :: A
  block_ss     :: B
end

function Gridap.Algebra.symbolic_setup(solver::BlockDiagonalSolver,mat::AbstractBlockMatrix)
  mat_blocks   = diag(blocks(mat))
  block_ss     = map(block_symbolic_setup,solver.blocks,solver.solvers,mat_blocks)
  return BlockDiagonalSolverSS(solver,block_ss)
end

function Gridap.Algebra.symbolic_setup(solver::BlockDiagonalSolver,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  mat_blocks   = diag(blocks(mat))
  block_ss     = map((b,m) -> block_symbolic_setup(b,m,x),solver.blocks,solver.solvers,mat_blocks)
  return BlockDiagonalSolverSS(solver,block_ss)
end

# Numerical setup

struct BlockDiagonalSolverNS{A,B,C} <: Gridap.Algebra.NumericalSetup
  solver       :: A
  block_ns     :: B
  work_caches  :: C
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalSolverSS,mat::AbstractBlockMatrix)
  solver     = ss.solver
  mat_blocks = diag(blocks(mat))
  block_ns   = map(block_numerical_setup,ss.block_ss,mat_blocks)

  y = mortar(map(allocate_in_domain,block_ns)); fill!(y,0.0)
  work_caches = y
  return BlockDiagonalSolverNS(solver,block_ns,work_caches)
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalSolverSS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver     = ss.solver
  mat_blocks = diag(blocks(mat))
  block_ns   = map((b,m) -> block_numerical_setup(b,m,x),ss.block_ss,mat_blocks)

  y = mortar(map(allocate_in_domain,block_ns)); fill!(y,0.0)
  work_caches = y
  return BlockDiagonalSolverNS(solver,block_ns,work_caches)
end

function Gridap.Algebra.numerical_setup!(ns::BlockDiagonalSolverNS,mat::AbstractBlockMatrix)
  mat_blocks   = diag(blocks(mat))
  map(block_numerical_setup!,ns.block_ns,mat_blocks)
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::BlockDiagonalSolverNS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  mat_blocks   = diag(blocks(mat))
  map((b,m) -> block_numerical_setup!(b,m,x),ns.block_ns,mat_blocks)
  return ns
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockDiagonalSolverNS,b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  y = ns.work_caches

  for (iB,bns) in enumerate(ns.block_ns)
    xi = blocks(x)[iB]
    bi = blocks(b)[iB]
    yi = blocks(y)[iB]
    solve!(yi,bns.ns,bi)
    copy!(xi,yi)
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockDiagonalSolverNS,b)
  solve!(x,ns,b)
end
