
abstract type SolverBlock end
abstract type LinearSolverBlock <: SolverBlock end
abstract type NonlinearSolverBlock <: SolverBlock end

struct MatrixBlock{A} <: LinearSolverBlock
  mat :: A
  function MatrixBlock(mat::AbstractMatrix)
    A = typeof(mat)
    return new{A}(mat)
  end
end

struct LinearSystemBlock <: LinearSolverBlock end
struct NonlinearSystemBlock <: NonlinearSolverBlock end

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

# Instantiate blocks

function instantiate_block_cache(block::LinearSolverBlock,mat::AbstractMatrix)
  @abstractmethod
end
function instantiate_block_cache(block::NonlinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
function instantiate_block_cache(block::LinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  instantiate_block_cache(block,mat)
end

function instantiate_block_cache(block::MatrixBlock,mat::AbstractMatrix)
  return block.mat
end
function instantiate_block_cache(block::BiformBlock,mat::AbstractMatrix)
  return assemble_matrix(block.f,block.assem,block.trial,block.test)
end
instantiate_block_cache(block::LinearSystemBlock,mat::AbstractMatrix) = mat

function instantiate_block_cache(block::TriformBlock,mat::AbstractMatrix,x::AbstractVector)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  return assemble_matrix(f,block.assem,block.trial,block.test)
end
instantiate_block_cache(block::NonlinearSystemBlock,mat::AbstractMatrix,x::AbstractVector) = mat

# Update blocks

function update_block_cache!(cache,block::LinearSolverBlock,mat::AbstractMatrix)
  return cache
end
function update_block_cache!(cache,block::NonlinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
function update_block_cache!(cache,block::LinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  update_block_cache!(cache,block,mat)
end

function update_block_cache!(cache,block::TriformBlock,mat::AbstractMatrix,x::AbstractVector)
  uh = FEFunction(block.trial,x)
  f(u,v) = block.f(uh,u,v)
  assemble_matrix!(mat,f,block.assem,block.trial,block.test)
end
function update_block_cache!(cache,block::NonlinearSystemBlock,mat::AbstractMatrix,x::AbstractVector)
  return cache
end
