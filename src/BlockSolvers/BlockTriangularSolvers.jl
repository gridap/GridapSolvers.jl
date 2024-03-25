"""
    struct BlockTriangularSolver <: Gridap.Algebra.LinearSolver

  Solver representing a block-triangular (upper/lower) solver, i.e 

  [ A11  c12⋅A12  c13⋅A13 ] [ x1 ] = [ r1 ]
  [  0       A22  c23⋅A23 ] [ x2 ] = [ r2 ]
  [  0      0       A33   ] [ x3 ] = [ r3 ]

  # Parameters: 
  - `blocks::AbstractMatrix{<:SolverBlock}`: Matrix of solver blocks, indicating how 
      each block of the preconditioner is obtained. 
  - `solvers::AbstractVector{<:Gridap.Algebra.LinearSolver}`: Vector of solvers, 
      one for each diagonal block.
  - `coeffs::AbstractMatrix{<:Real}`: Matrix of coefficients, indicating the 
      contribution of the off-diagonal blocks to the right-hand side of each 
      diagonal. In particular, blocks can be turned off by setting the corresponding 
      coefficient to zero.
  - `half::Symbol`: Either `:upper` or `:lower`.
"""
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
  
  y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0) # This should be removed with PA 0.4
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w, y
  return BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
end

function Gridap.Algebra.numerical_setup(ss::BlockTriangularSolverSS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver      = ss.solver
  block_ns    = map(numerical_setup,ss.block_ss,diag(ss.block_caches),blocks(x))

  y = mortar(map(allocate_in_domain,diag(ss.block_caches))); fill!(y,0.0)
  w = allocate_in_range(mat); fill!(w,0.0)
  work_caches = w, y
  return BlockTriangularSolverNS(solver,block_ns,ss.block_caches,work_caches)
end

function Gridap.Algebra.numerical_setup!(ns::BlockTriangularSolverNS,mat::AbstractBlockMatrix)
  solver       = ns.solver
  mat_blocks   = blocks(mat)
  block_caches = map(update_block_cache!,ns.block_caches,solver.blocks,mat_blocks)
  map(diag(solver.blocks),ns.block_ns,diag(block_caches)) do bi, nsi, ci
    if is_nonlinear(bi)
      numerical_setup!(nsi,ci)
    end
  end
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::BlockTriangularSolverNS,mat::AbstractBlockMatrix,x::AbstractBlockVector)
  solver       = ns.solver
  mat_blocks   = blocks(mat)
  vec_blocks   = blocks(x)
  block_caches = map(CartesianIndices(solver.blocks)) do I
    update_block_cache!(ns.block_caches[I],solver.blocks[I],mat_blocks[I],vec_blocks[I[2]])
  end
  map(diag(solver.blocks),ns.block_ns,diag(block_caches),vec_blocks) do bi, nsi, ci, xi
    if is_nonlinear(bi)
      numerical_setup!(nsi,ci,xi)
    end
  end
  return ns
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockTriangularSolverNS{Val{:lower}},b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  NB = length(ns.block_ns)
  c = ns.solver.coeffs
  w, y = ns.work_caches
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
    yi  = y[Block(iB)]
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
    xi  = x[Block(iB)]
    yi  = y[Block(iB)]
    solve!(yi,nsi,wi)
    copy!(xi,yi) # Remove this with PA 0.4
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockTriangularSolverNS,b)
  solve!(x,ns,b)
end
