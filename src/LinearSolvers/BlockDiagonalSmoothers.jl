struct BlockDiagonalSmoother{A,B,C} <: Gridap.Algebra.LinearSolver
  num_blocks :: Int32
  ranges     :: A
  blocks     :: B
  solvers    :: C

  function BlockDiagonalSmoother(ranges,blocks,solvers)
    num_blocks = length(ranges)
    @check length(blocks)  == num_blocks
    @check length(solvers) == num_blocks

    A = typeof(ranges)
    B = typeof(blocks)
    C = typeof(solvers)
    return new{A,B,C}(num_blocks,ranges,blocks,solvers)
  end
end

function BlockDiagonalSmoother(biforms :: AbstractArray{<:Function},
                               trials  :: AbstractArray{<:FESpace},
                               tests   :: AbstractArray{<:FESpace},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  ranges = map(num_free_dofs,tests)
  blocks = compute_block_matrices(biforms,trials,tests)
  return BlockDiagonalSmoother(ranges,blocks,solvers)
end

function BlockDiagonalSmoother(biforms :: AbstractArray{<:Function},
                               U       :: MultiFieldFESpace,
                               V       :: MultiFieldFESpace,
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver})
  dof_ids = get_free_dof_ids(V)
  ranges  = map(i->dof_ids[Block(i)],1:blocklength(dof_ids))
  blocks  = compute_block_matrices(biforms,U.spaces,V.spaces)
  return BlockDiagonalSmoother(ranges,blocks,solvers)
end

function BlockDiagonalSmoother(A       :: AbstractMatrix,
                               ranges  :: AbstractArray{<:AbstractRange},
                               solvers :: AbstractArray{<:Gridap.Algebra.LinearSolver};
                               lazy_mode=false)
  blocks = extract_diagonal_blocks(A,ranges;lazy_mode=lazy_mode)
  return BlockDiagonalSmoother(ranges,blocks,solvers)
end

function compute_block_matrices(biforms :: AbstractArray{<:Function},
                                trials  :: AbstractArray{<:FESpace},
                                tests   :: AbstractArray{<:FESpace})
  @check length(biforms) == length(tests) == length(trials)
  @check all(U -> isa(U,TrialFESpace),trials)

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

# TODO: Should we consider overlapping block smoothers? 
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

function LinearAlgebra.ldiv!(x,ns::BlockDiagonalSmootherNS,b)
  solve!(x,ns,b)
end