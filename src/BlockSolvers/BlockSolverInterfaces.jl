
abstract type SolverBlock end

abstract type LinearSolverBlock <: SolverBlock end
abstract type NonlinearSolverBlock <: SolverBlock end

is_nonlinear(::LinearSolverBlock) = false
is_nonlinear(::NonlinearSolverBlock) = true

function instantiate_block_cache(block::LinearSolverBlock,mat::AbstractMatrix)
  @abstractmethod
end
function instantiate_block_cache(block::NonlinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
function instantiate_block_cache(block::LinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  instantiate_block_cache(block,mat)
end

function update_block_cache!(cache,block::LinearSolverBlock,mat::AbstractMatrix)
  return cache
end
function update_block_cache!(cache,block::NonlinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
function update_block_cache!(cache,block::LinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  update_block_cache!(cache,block,mat)
end

# MatrixBlock

struct MatrixBlock{A} <: LinearSolverBlock
  mat :: A
  function MatrixBlock(mat::AbstractMatrix)
    A = typeof(mat)
    return new{A}(mat)
  end
end

instantiate_block_cache(block::MatrixBlock,::AbstractMatrix) = block.mat

# SystemBlocks

struct LinearSystemBlock <: LinearSolverBlock end
struct NonlinearSystemBlock <: NonlinearSolverBlock end

instantiate_block_cache(block::LinearSystemBlock,mat::AbstractMatrix) = mat
instantiate_block_cache(block::NonlinearSystemBlock,mat::AbstractMatrix,::AbstractVector) = mat
update_block_cache!(cache,block::NonlinearSystemBlock,mat::AbstractMatrix,::AbstractVector) = mat

# BiformBlock/TriformBlock
struct BiformBlock <: LinearSolverBlock
  f     :: Function
  trial :: FESpace
  test  :: FESpace
  assem :: Assembler
  function BiformBlock(f::Function,
                       trial::FESpace,
                       test::FESpace,
                       assem=SparseMatrixAssembler(trial,test))
    return new(f,trial,test,assem)
  end
end

struct TriformBlock <: NonlinearSolverBlock
  f     :: Function
  trial :: FESpace
  test  :: FESpace
  assem :: Assembler
  function TriformBlock(f::Function,
                        trial::FESpace,
                        test::FESpace,
                        assem=SparseMatrixAssembler(trial,test))
    return new(f,trial,test,assem)
  end
end

function instantiate_block_cache(block::BiformBlock,mat::AbstractMatrix)
  return assemble_matrix(block.f,block.assem,block.trial,block.test)
end

function instantiate_block_cache(block::TriformBlock,mat::AbstractMatrix,x::AbstractVector)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  return assemble_matrix(f,block.assem,block.trial,block.test)
end

function update_block_cache!(cache,block::TriformBlock,mat::AbstractMatrix,x::AbstractVector)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  assemble_matrix!(mat,f,block.assem,block.trial,block.test)
end

# CompositeBlock
# How do we deal with different sparsity patterns? Not trivial...
