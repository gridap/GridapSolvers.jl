struct BlockDiagonalSmoother{A,B,C} <: Gridap.Algebra.LinearSolver
  num_blocks :: Int32
  ranges     :: A
  blocks     :: B
  solvers    :: C

  function BlockDiagonalSmoother(ranges,blocks,solvers)
    num_blocks = length(blocks)

    A = typeof(ranges)
    B = typeof(blocks)
    C = typeof(solvers)
    return new{A,B,C}(num_blocks,ranges,blocks,solvers)
  end
end

# Constructors 

function BlockDiagonalSmoother(blocks  :: AbstractArray{<:AbstractMatrix},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  ranges = compute_block_ranges(blocks...)
  return BlockDiagonalSmoother(ranges,blocks,solvers)
end

function BlockDiagonalSmoother(block_mat :: BlockMatrix,
                               solvers   :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  blocks = [block_mat[Block(i,i)] for i in 1:length(solvers)]
  ranges = compute_block_ranges(blocks...)
  return BlockDiagonalSmoother(ranges,blocks,solvers)
end

function BlockDiagonalSmoother(biforms :: AbstractArray{<:Function},
                               trials  :: AbstractArray{<:FESpace},
                               tests   :: AbstractArray{<:FESpace},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  blocks = compute_block_matrices(biforms,trials,tests)
  return BlockDiagonalSmoother(blocks,solvers)
end

function BlockDiagonalSmoother(biforms :: AbstractArray{<:Function},
                               U       :: Union{MultiFieldFESpace,GridapDistributed.DistributedMultiFieldFESpace},
                               V       :: Union{MultiFieldFESpace,GridapDistributed.DistributedMultiFieldFESpace},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  return BlockDiagonalSmoother(biforms,[U...],[V...],solvers)
end

function BlockDiagonalSmoother(A       :: AbstractMatrix,
                               ranges  :: AbstractArray{<:AbstractRange},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver};
                               lazy_mode=false)
  blocks = extract_diagonal_blocks(A,ranges;lazy_mode=lazy_mode)
  return BlockDiagonalSmoother(ranges,blocks,solvers)
end

# Computing blocks and ranges

function compute_block_ranges(blocks::AbstractMatrix...)
  num_blocks = length(blocks)
  ranges     = Vector{AbstractRange}(undef,num_blocks)
  ranges[1]  = 1:size(blocks[1],2)
  for i in 2:num_blocks
    ranges[i] = size(blocks[i-1],2) .+ (1:size(blocks[i],2))
  end
  return ranges
end

function compute_block_ranges(blocks::PSparseMatrix...)
  _blocks = map(b -> own_values(b),blocks)
  ranges = map(_blocks...) do blocks...
    compute_block_ranges(blocks...)
  end
  return ranges
end

function compute_block_matrices(biforms :: AbstractArray{<:Function},
                                trials  :: AbstractArray{<:FESpace},
                                tests   :: AbstractArray{<:FESpace})
  @check length(biforms) == length(tests) == length(trials)

  blocks = map(assemble_matrix,biforms,tests,trials)
  return blocks
end

function extract_diagonal_blocks(A::AbstractMatrix,ranges;lazy_mode=false)
  blocks = map(ranges) do range
    if lazy_mode
      view(A,range,range)
    else
      A[range,range]
    end
  end
  return blocks
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

struct BlockDiagonalSmootherNS{A,B,C} <: Gridap.Algebra.NumericalSetup
  solver   :: A
  block_ns :: B
  caches   :: C
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalSmootherSS,mat::AbstractMatrix)
  solver   = ss.solver
  block_ns = map(numerical_setup,ss.block_ss,solver.blocks)
  caches   = _get_block_diagonal_smoothers_caches(solver.blocks,mat)
  return BlockDiagonalSmootherNS(solver,block_ns,caches)
end

function _get_block_diagonal_smoothers_caches(blocks,mat)
  return nothing
end

function _get_block_diagonal_smoothers_caches(blocks::AbstractArray{<:PSparseMatrix},mat::PSparseMatrix)
  x_blocks = map(bi->allocate_col_vector(bi),blocks)
  b_blocks = map(bi->allocate_col_vector(bi),blocks)
  return x_blocks,b_blocks
end

function _get_block_diagonal_smoothers_caches(blocks::AbstractArray{<:PSparseMatrix},mat::BlockMatrix)
  return nothing
end

# Solve

function to_blocks!(x::AbstractVector,x_blocks,ranges)
  map(ranges,x_blocks) do range,x_block
    x_block .= x[range]
  end
  return x_blocks
end

# TODO: The exchange could be optimized for sure by swapping the loop order...
function to_blocks!(x::PVector,x_blocks,ranges)
  x_blocks_owned = map(xi->own_values(xi),x_blocks)
  map(own_values(x),ranges,x_blocks_owned...) do x,ranges,x_blocks...
    to_blocks!(x,x_blocks,ranges)
  end
  map(x_blocks) do x
    consistent!(x) |> fetch
  end
  return x_blocks
end

function to_global!(x::AbstractVector,x_blocks,ranges)
  map(ranges,x_blocks) do range,x_block
    x[range] .= x_block
  end
  return x
end

function to_global!(x::PVector,x_blocks,ranges)
  x_blocks_owned = map(xi->own_values(xi),x_blocks)
  map(own_values(x),ranges,x_blocks_owned...) do x,ranges,x_blocks...
    to_global!(x,x_blocks,ranges)
  end
  consistent!(x) |> fetch
  return x
end

# Solve for serial vectors
function Gridap.Algebra.solve!(x::AbstractVector,ns::BlockDiagonalSmootherNS,b::AbstractVector)
  solver, block_ns = ns.solver, ns.block_ns
  num_blocks, ranges = solver.num_blocks, solver.ranges
  for iB in 1:num_blocks
    xi = view(x,ranges[iB])
    bi = view(b,ranges[iB])
    solve!(xi,block_ns[iB],bi)
  end
  return x
end

# Solve for PVectors (parallel)
function Gridap.Algebra.solve!(x::PVector,ns::BlockDiagonalSmootherNS,b::PVector)
  solver, block_ns, caches = ns.solver, ns.block_ns, ns.caches
  num_blocks, ranges = solver.num_blocks, solver.ranges
  x_blocks, b_blocks = caches

  to_blocks!(x,x_blocks,ranges)
  to_blocks!(b,b_blocks,ranges)
  for iB in 1:num_blocks
    xi = x_blocks[iB]
    bi = b_blocks[iB]
    solve!(xi,block_ns[iB],bi)
  end
  to_global!(x,x_blocks,ranges)
  return x
end

# Solve for BlockVectors (serial & parallel)
function Gridap.Algebra.solve!(x::BlockVector,ns::BlockDiagonalSmootherNS,b::BlockVector)
  solver, block_ns = ns.solver, ns.block_ns
  num_blocks = solver.num_blocks

  @check blocklength(x) == blocklength(b) == num_blocks
  for iB in 1:num_blocks
    xi = x[Block(iB)]
    bi = b[Block(iB)]
    solve!(xi,block_ns[iB],bi)
  end

  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockDiagonalSmootherNS,b)
  solve!(x,ns,b)
end