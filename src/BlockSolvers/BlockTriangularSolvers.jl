"""
    struct BlockTriangularSolver{T,N} <: LinearSolver

Solver representing a block-triangular (upper/lower) solver, i.e 

```
│ A11  c12⋅A12  c13⋅A13 │   │ x1 │   │ r1 │
│  0       A22  c23⋅A23 │ ⋅ │ x2 │ = │ r2 │
│  0      0         A33 │   │ x3 │   │ r3 │
```

where `N` is the number of diagonal blocks and `T` is either `Val{:upper}` or `Val{:lower}`.

# Properties: 

- `blocks::AbstractMatrix{<:SolverBlock}`: Matrix of solver blocks, indicating how 
    each block of the preconditioner is obtained. 
- `solvers::AbstractVector{<:LinearSolver}`: Vector of solvers, 
    one for each diagonal block.
- `coeffs::AbstractMatrix{<:Real}`: Matrix of coefficients, indicating the 
    contribution of the off-diagonal blocks to the right-hand side of each 
    diagonal. In particular, blocks can be turned off by setting the corresponding 
    coefficient to zero.

"""
struct BlockTriangularSolver{T,N,A,B,C} <: Gridap.Algebra.LinearSolver
  blocks  :: A
  solvers :: B
  coeffs  :: C

  @doc """
      function BlockTriangularSolver(
        blocks  :: AbstractMatrix{<:SolverBlock},
        solvers :: AbstractVector{<:LinearSolver},
        coeffs = fill(1.0,size(blocks)),
        half   = :upper
      )
    
  Create and instance of [`BlockTriangularSolver`](@ref) from its underlying properties.
  The kwarg `half` can have values `:upper` or `:lower`.
  """
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

@doc """
    function BlockTriangularSolver(
      solvers::AbstractVector{<:LinearSolver}; 
      is_nonlinear::Matrix{Bool}=fill(false,(length(solvers),length(solvers))),
      coeffs=fill(1.0,size(is_nonlinear)),
      half=:upper
    )

Create and instance of [`BlockTriangularSolver`](@ref) where all blocks are extracted from 
the linear system.
The kwarg `half` can have values `:upper` or `:lower`.
"""
function BlockTriangularSolver(
  solvers::AbstractVector{<:Gridap.Algebra.LinearSolver}; 
  is_nonlinear::Matrix{Bool}=fill(false,(length(solvers),length(solvers))),
  coeffs=fill(1.0,size(is_nonlinear)),
  half=:upper
)
  blocks = map(nl -> nl ? NonlinearSystemBlock() : LinearSystemBlock(),is_nonlinear)
  return BlockTriangularSolver(blocks,solvers,coeffs,half)
end

# Symbolic setup

struct BlockTriangularSolverSS{A,B,C} <: Gridap.Algebra.SymbolicSetup
  solver    :: A
  block_ss  :: B
  block_off :: C
end

function Gridap.Algebra.symbolic_setup(solver::BlockTriangularSolver,mat::AbstractBlockMatrix)
  mat_blocks = blocks(mat)
  block_ss   = map(block_symbolic_setup,diag(solver.blocks),solver.solvers,diag(mat_blocks))
  block_off  = map(CartesianIndices(mat_blocks)) do I
    if I[1] != I[2]
      block_offdiagonal_setup(solver.blocks[I],mat_blocks[I])
    else
      mat_blocks[I]
    end
  end
  return BlockTriangularSolverSS(solver,block_ss,block_off)
end

function Gridap.Algebra.symbolic_setup(solver::BlockTriangularSolver{T,N},mat::AbstractBlockMatrix,x::AbstractBlockVector) where {T,N}
  mat_blocks = blocks(mat)
  block_ss   = map((b,s,m) -> block_symbolic_setup(b,s,m,x),diag(solver.blocks),solver.solvers,diag(mat_blocks))
  block_off  = map(CartesianIndices(mat_blocks)) do I
    if I[1] != I[2]
      block_offdiagonal_setup(solver.blocks[I],mat_blocks[I],x)
    else
      mat_blocks[I]
    end
  end
  return BlockTriangularSolverSS(solver,block_ss,block_off)
end

# Numerical setup

struct BlockTriangularSolverNS{T,A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver      :: A
  block_ns    :: B
  block_off   :: C
  work_caches :: D
  function BlockTriangularSolverNS(
    solver::BlockTriangularSolver{T},
    block_ns,block_off,work_caches
  ) where T
    A = typeof(solver) 
    B = typeof(block_ns)
    C = typeof(block_off)
    D = typeof(work_caches)
    return new{T,A,B,C,D}(solver,block_ns,block_off,work_caches)
  end
end

function Gridap.Algebra.numerical_setup(ss::BlockTriangularSolverSS,mat::AbstractBlockMatrix)
  solver   = ss.solver
  block_ns = map(block_numerical_setup,ss.block_ss,diag(blocks(mat)))
  
  y = mortar(map(allocate_in_domain,block_ns)); fill!(y,0.0) # This should be removed with PA 0.4
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w, y
  return BlockTriangularSolverNS(solver,block_ns,ss.block_off,work_caches)
end

function Gridap.Algebra.numerical_setup(ss::BlockTriangularSolverSS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver     = ss.solver
  mat_blocks = blocks(mat)
  block_ns   = map((b,m) -> block_numerical_setup(b,m,x),ss.block_ss,diag(mat_blocks))

  y = mortar(map(allocate_in_domain,block_ns)); fill!(y,0.0)
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w, y
  return BlockTriangularSolverNS(solver,block_ns,ss.block_off,work_caches)
end

function Gridap.Algebra.numerical_setup!(ns::BlockTriangularSolverNS,mat::AbstractBlockMatrix)
  solver       = ns.solver
  mat_blocks   = blocks(mat)
  map(ns.block_ns,diag(mat_blocks)) do nsi, mi
    if is_nonlinear(nsi)
      block_numerical_setup!(nsi,mi)
    end
  end
  map(CartesianIndices(mat_blocks)) do I
    if (I[1] != I[2]) && is_nonlinear(solver.blocks[I])
      block_offdiagonal_setup!(ns.block_off[I],solver.blocks[I],mat_blocks[I])
    end
  end
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::BlockTriangularSolverNS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver       = ns.solver
  mat_blocks   = blocks(mat)
  map(ns.block_ns,diag(mat_blocks)) do nsi, mi
    if is_nonlinear(nsi)
      block_numerical_setup!(nsi,mi,x)
    end
  end
  map(CartesianIndices(mat_blocks)) do I
    if (I[1] != I[2]) && is_nonlinear(solver.blocks[I])
      block_offdiagonal_setup!(ns.block_off[I],solver.blocks[I],mat_blocks[I],x)
    end
  end
  return ns
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockTriangularSolverNS{Val{:lower}},b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w, y = ns.work_caches
  mats = ns.block_off
  for iB in 1:NB
    # Add lower off-diagonal contributions
    wi  = blocks(w)[iB]
    copy!(wi,blocks(b)[iB])
    for jB in 1:iB-1
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = blocks(x)[jB]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB].ns
    xi  = blocks(x)[iB]
    yi  = blocks(y)[iB]
    solve!(yi,nsi,wi)
    copy!(xi,yi)
  end
  return x
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockTriangularSolverNS{Val{:upper}},b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w, y = ns.work_caches
  mats = ns.block_off
  for iB in NB:-1:1
    # Add upper off-diagonal contributions
    wi  = blocks(w)[iB]
    copy!(wi,blocks(b)[iB])
    for jB in iB+1:NB
      cij = c[iB,jB]
      if abs(cij) > eps(cij)
        xj = blocks(x)[jB]
        mul!(wi,mats[iB,jB],xj,-cij,1.0)
      end
    end

    # Solve diagonal block
    nsi = ns.block_ns[iB].ns
    xi  = blocks(x)[iB]
    yi  = blocks(y)[iB]
    solve!(yi,nsi,wi)
    copy!(xi,yi) # Remove this with PA 0.4
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockTriangularSolverNS,b)
  solve!(x,ns,b)
end
