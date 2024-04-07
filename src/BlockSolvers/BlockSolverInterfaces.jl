
"""
    abstract type SolverBlock end

  Abstract type representing a block in a block solver. More specifically, it 
  indicates how a block is obtained from the original system matrix.
"""
abstract type SolverBlock end

"""
    abstract type LinearSolverBlock <: SolverBlock end

  SolverBlock that will not be updated between nonlinear iterations.
"""
abstract type LinearSolverBlock <: SolverBlock end

"""
    abstract type NonlinearSolverBlock <: SolverBlock end

  SolverBlock that will be updated between nonlinear iterations.
"""
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
  return cache
end

# MatrixBlock

"""
    struct MatrixBlock{A} <: LinearSolverBlock
  
  SolverBlock representing an external, independent matrix. 
  
  # Parameters: 
  - `mat::A`: The matrix.
"""
struct MatrixBlock{A} <: LinearSolverBlock
  mat :: A
  function MatrixBlock(mat::AbstractMatrix)
    A = typeof(mat)
    return new{A}(mat)
  end
end

instantiate_block_cache(block::MatrixBlock,::AbstractMatrix) = block.mat

# SystemBlocks

"""
    struct LinearSystemBlock <: LinearSolverBlock

  SolverBlock representing a linear (i.e non-updateable) block that is directly 
  taken from the system matrix. This block will not be updated between nonlinear 
  iterations.
"""
struct LinearSystemBlock <: LinearSolverBlock end

"""
    struct NonlinearSystemBlock <: LinearSolverBlock

  SolverBlock representing a nonlinear (i.e updateable) block that is directly 
  taken from the system matrix. This block will be updated between nonlinear
  iterations.
"""
struct NonlinearSystemBlock <: NonlinearSolverBlock end

instantiate_block_cache(block::LinearSystemBlock,mat::AbstractMatrix) = mat
instantiate_block_cache(block::NonlinearSystemBlock,mat::AbstractMatrix,::AbstractVector) = mat
update_block_cache!(cache,block::NonlinearSystemBlock,mat::AbstractMatrix,::AbstractVector) = mat

# BiformBlock/TriformBlock
"""
    struct BiformBlock <: LinearSolverBlock

  SolverBlock representing a linear block assembled from a bilinear form. 
  This block will be not updated between nonlinear iterations.
  
  # Parameters: 
  - `f::Function`: The bilinear form, i.e f(du,dv) = ∫(...)dΩ
  - `trial::FESpace`: The trial space.
  - `test::FESpace`: The test space.
  - `assem::Assembler`: The assembler to use.
"""
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

"""
    struct TriformBlock <: NonlinearSolverBlock

  SolverBlock representing a nonlinear block assembled from a trilinear form. 
  This block will be updated between nonlinear iterations.
  
  # Parameters: 
  - `f::Function`: The trilinear form, i.e f(u,du,dv) = ∫(...)dΩ
  - `trial::FESpace`: The trial space.
  - `test::FESpace`: The test space.
  - `assem::Assembler`: The assembler to use.

"""
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
  return assemble_matrix!(f,cache,block.assem,block.trial,block.test)
end

# CompositeBlock, i.e something that takes from the system matrix and adds another contribution to it.
# How do we deal with different sparsity patterns? Not trivial...
