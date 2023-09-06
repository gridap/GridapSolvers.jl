struct BlockDiagonalSmoother{A,B} <: Gridap.Algebra.LinearSolver
  blocks  :: A
  solvers :: B
  function BlockDiagonalSmoother(blocks ::AbstractArray{<:AbstractMatrix},
                                 solvers::AbstractArray{<:Gridap.Algebra.LinearSolver})
    @check length(blocks) == length(solvers)
    A = typeof(blocks)
    B = typeof(solvers)
    return new{A,B}(blocks,solvers)
  end
end

# Constructors 

function BlockDiagonalSmoother(block_mat :: AbstractBlockMatrix,
                               solvers   :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  mat_blocks = diag(blocks(block_mat))
  return BlockDiagonalSmoother(mat_blocks,solvers)
end

function BlockDiagonalSmoother(biforms :: AbstractArray{<:Function},
                               trials  :: AbstractArray{<:FESpace},
                               tests   :: AbstractArray{<:FESpace},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  mat_blocks = compute_block_matrices(biforms,trials,tests)
  return BlockDiagonalSmoother(mat_blocks,solvers)
end

function BlockDiagonalSmoother(biforms :: AbstractArray{<:Function},
                               U       :: Union{MultiFieldFESpace,GridapDistributed.DistributedMultiFieldFESpace},
                               V       :: Union{MultiFieldFESpace,GridapDistributed.DistributedMultiFieldFESpace},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  return BlockDiagonalSmoother(biforms,[U...],[V...],solvers)
end

function compute_block_matrices(biforms :: AbstractArray{<:Function},
                                trials  :: AbstractArray{<:FESpace},
                                tests   :: AbstractArray{<:FESpace})
  @check length(biforms) == length(tests) == length(trials)
  mat_blocks = map(assemble_matrix,biforms,tests,trials)
  return mat_blocks
end

# Symbolic and numerical setup
struct BlockDiagonalSmootherSS{A,B} <: Gridap.Algebra.SymbolicSetup
  solver   :: A
  block_ss :: B
end

function Gridap.Algebra.symbolic_setup(solver::BlockDiagonalSmoother,mat::AbstractMatrix)
  block_ss = map(symbolic_setup,solver.solvers,solver.blocks)
  return BlockDiagonalSmootherSS(solver,block_ss)
end

struct BlockDiagonalSmootherNS{A,B} <: Gridap.Algebra.NumericalSetup
  solver   :: A
  block_ns :: B
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalSmootherSS,mat::AbstractMatrix)
  solver   = ss.solver
  block_ns = map(numerical_setup,ss.block_ss,solver.blocks)
  return BlockDiagonalSmootherNS(solver,block_ns)
end

# Solve

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockDiagonalSmootherNS,b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  for (iB,bns) in enumerate(ns.block_ns)
    xi = x[Block(iB)]
    bi = b[Block(iB)]
    solve!(xi,bns,bi)
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockDiagonalSmootherNS,b)
  solve!(x,ns,b)
end