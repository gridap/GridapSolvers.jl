
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

struct BlockSS{A,B,C} <: Algebra.SymbolicSetup 
  block :: A
  ss    :: B
  cache :: C
end

mutable struct BlockNS{A,B,C,D} <: Algebra.NumericalSetup
  block :: A
  ns    :: B
  mat   :: C
  cache :: D
end

is_nonlinear(ss::BlockSS) = is_nonlinear(ss.block)
is_nonlinear(ns::BlockNS) = is_nonlinear(ns.block)

function Algebra.allocate_in_domain(block::BlockNS)
  return allocate_in_domain(block.mat)
end
function Algebra.allocate_in_range(block::BlockNS)
  return allocate_in_range(block.mat)
end

function restrict_blocks(x::AbstractBlockVector,ids::Vector{Int8})
  if isempty(ids)
    return x
  elseif length(ids) == 1
    return blocks(x)[ids[1]]
  else
    return mortar(blocks(x)[ids])
  end
end

"""
    block_symbolic_setup(block::SolverBlock,solver::LinearSolver,mat::AbstractMatrix)
    block_symbolic_setup(block::SolverBlock,solver::LinearSolver,mat::AbstractMatrix,x::AbstractVector)

  Given a SolverBlock, returns the symbolic setup associated to the LinearSolver.
"""
function block_symbolic_setup(block::SolverBlock,solver::LinearSolver,mat::AbstractMatrix)
  @abstractmethod
end
function block_symbolic_setup(block::SolverBlock,solver::LinearSolver,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
@inline function block_symbolic_setup(block::LinearSolverBlock,solver::LinearSolver,mat::AbstractMatrix,x::AbstractVector)
  return block_symbolic_setup(block,solver,mat)
end

"""
    block_numerical_setup(block::SolverBlock,ss::BlockSS,mat::AbstractMatrix)
    block_numerical_setup(block::SolverBlock,ss::BlockSS,mat::AbstractMatrix,x::AbstractVector)

  Given a SolverBlock, returns the numerical setup associated to it.
"""
function block_numerical_setup(block::SolverBlock,ss::BlockSS,mat::AbstractMatrix)
  @abstractmethod
end
function block_numerical_setup(block::SolverBlock,ss::BlockSS,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
@inline function block_numerical_setup(block::LinearSolverBlock,ss::BlockSS,mat::AbstractMatrix,x::AbstractVector)
  return block_numerical_setup(block,ss,mat)
end
@inline function block_numerical_setup(ss::BlockSS,mat::AbstractMatrix)
  block_numerical_setup(ss.block,ss,mat)
end
@inline function block_numerical_setup(ss::BlockSS,mat::AbstractMatrix,x::AbstractVector)
  block_numerical_setup(ss.block,ss,mat,x)
end

"""
    block_numerical_setup!(block::SolverBlock,ns::BlockNS,mat::AbstractMatrix)
    block_numerical_setup!(block::SolverBlock,ns::BlockNS,mat::AbstractMatrix,x::AbstractVector)

  Given a SolverBlock, updates the numerical setup associated to it.
"""
function block_numerical_setup!(block::LinearSolverBlock,ns::BlockNS,mat::AbstractMatrix)
  ns
end
function block_numerical_setup!(block::LinearSolverBlock,ns::BlockNS,mat::AbstractMatrix,x::AbstractVector)
  ns
end
function block_numerical_setup!(block::NonlinearSolverBlock,ns::BlockNS,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
@inline function block_numerical_setup!(ns::BlockNS,mat::AbstractMatrix)
  block_numerical_setup!(ns.block,ns,mat)
end
@inline function block_numerical_setup!(ns::BlockNS,mat::AbstractMatrix,x::AbstractVector)
  block_numerical_setup!(ns.block,ns,mat,x)
end

"""
    block_offdiagonal_setup(block::SolverBlock,mat::AbstractMatrix)
    block_offdiagonal_setup(block::SolverBlock,mat::AbstractMatrix,x::AbstractVector)

  Given a SolverBlock, returns the off-diagonal block of associated to it.
"""
function block_offdiagonal_setup(block::SolverBlock,mat::AbstractMatrix)
  @abstractmethod
end
function block_offdiagonal_setup(block::NonlinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
end
@inline function block_offdiagonal_setup(block::LinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  return block_offdiagonal_setup(block,mat)
end

"""
    block_offdiagonal_setup!(cache,block::SolverBlock,mat::AbstractMatrix)
    block_offdiagonal_setup!(cache,block::SolverBlock,mat::AbstractMatrix,x::AbstractVector)

  Given a SolverBlock, updates the off-diagonal block of associated to it.
"""
@inline function block_offdiagonal_setup!(cache,block::LinearSolverBlock,mat::AbstractMatrix)
  return cache
end
@inline function block_offdiagonal_setup!(cache,block::LinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  return cache
end
function block_offdiagonal_setup!(cache,block::NonlinearSolverBlock,mat::AbstractMatrix,x::AbstractVector)
  @abstractmethod
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

function block_symbolic_setup(block::MatrixBlock,solver::LinearSolver,mat::AbstractMatrix)
  return BlockSS(block,symbolic_setup(solver,block.mat),nothing)
end

function block_numerical_setup(block::MatrixBlock,ss::BlockSS,mat::AbstractMatrix)
  return BlockNS(block,numerical_setup(ss.ss,block.mat),block.mat,nothing)
end

function block_offdiagonal_setup(block::MatrixBlock,mat::AbstractMatrix)
  return block.mat
end

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
      I :: Vector{Int}
    end

  SolverBlock representing a nonlinear (i.e updateable) block that is directly 
  taken from the system matrix. This block will be updated between nonlinear
  iterations.

  # Parameters: 
    - `ids::Vector{Int8}`: Block indices considered when updating the nonlinear block. 
       By default, all indices are considered. 
"""
struct NonlinearSystemBlock <: NonlinearSolverBlock 
  ids :: Vector{Int8}
  function NonlinearSystemBlock(ids::Vector{<:Integer})
    new(convert(Vector{Int8},ids))
  end
end

NonlinearSystemBlock() = NonlinearSystemBlock(Int8[])
NonlinearSystemBlock(id::Integer) = NonlinearSystemBlock([Int8(id)])

function block_symbolic_setup(block::LinearSystemBlock,solver::LinearSolver,mat::AbstractMatrix)
  return BlockSS(block,symbolic_setup(solver,mat),mat)
end
function block_symbolic_setup(block::NonlinearSystemBlock,solver::LinearSolver,mat::AbstractMatrix,x::AbstractVector)
  y = restrict_blocks(x,block.ids)
  return BlockSS(block,symbolic_setup(solver,mat,y),mat)
end

function block_numerical_setup(block::LinearSystemBlock,ss::BlockSS,mat::AbstractMatrix)
  return BlockNS(block,numerical_setup(ss.ss,mat),mat,nothing)
end
function block_numerical_setup(block::NonlinearSystemBlock,ss::BlockSS,mat::AbstractMatrix,x::AbstractVector)
  y = restrict_blocks(x,block.ids)
  return BlockNS(block,numerical_setup(ss.ss,mat,y),mat,nothing)
end

function block_numerical_setup!(block::NonlinearSystemBlock,ns::BlockNS,mat::AbstractMatrix,x::AbstractVector)
  y = restrict_blocks(x,block.ids)
  numerical_setup!(ns.ns,mat,y)
  return ns
end

function block_offdiagonal_setup(block::LinearSystemBlock,mat::AbstractMatrix)
  return mat
end
function block_offdiagonal_setup(block::NonlinearSystemBlock,mat::AbstractMatrix,x::AbstractVector)
  return mat
end
function block_offdiagonal_setup!(cache,block::NonlinearSystemBlock,mat::AbstractMatrix,x::AbstractVector)
  return mat
end

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
  function BiformBlock(
    f::Function,
    trial::FESpace,
    test::FESpace,
    assem=SparseMatrixAssembler(trial,test)
  )
    return new(f,trial,test,assem)
  end
end

"""
    struct TriformBlock <: NonlinearSolverBlock

  SolverBlock representing a nonlinear block assembled from a trilinear form. 
  This block will be updated between nonlinear iterations.
  
  # Parameters: 
  - `f::Function`: The trilinear form, i.e f(u,du,dv) = ∫(...)dΩ
  - `param::FESpace`: The parameter space, where `u` lives.
  - `trial::FESpace`: The trial space, where `du` lives.
  - `test::FESpace`: The test space, where `dv` lives.
  - `assem::Assembler`: The assembler to use.
  - `ids::Vector{Int8}`: Block indices considered when updating the nonlinear block. 
      By default, all indices are considered.
"""
struct TriformBlock <: NonlinearSolverBlock
  f     :: Function
  param :: FESpace
  trial :: FESpace
  test  :: FESpace
  ids   :: Vector{Int8}
  assem :: Assembler
  function TriformBlock(
    f::Function,
    param::FESpace,
    trial::FESpace,
    test::FESpace,
    ids::Vector{<:Integer}=Int8[],
    assem=SparseMatrixAssembler(trial,test)
  )
    return new(f,param,trial,test,convert(Vector{Int8},ids),assem)
  end
end

function TriformBlock(
  f::Function,trial::FESpace,test::FESpace,ids::Vector{<:Integer},assem=SparseMatrixAssembler(trial,test)
)
  return TriformBlock(f,trial,trial,test,ids,assem)
end

function TriformBlock(
  f::Function,trial::FESpace,test::FESpace,id::Integer,assem=SparseMatrixAssembler(trial,test)
)
  return TriformBlock(f,trial,trial,test,[Int8(id)],assem)
end

function block_symbolic_setup(block::BiformBlock,solver::LinearSolver,mat::AbstractMatrix)
  A = assemble_matrix(block.f,block.assem,block.trial,block.test)
  return BlockSS(block,symbolic_setup(solver,A),A)
end
function block_numerical_setup(block::BiformBlock,ss::BlockSS,mat::AbstractMatrix)
  A = ss.cache
  return BlockNS(block,numerical_setup(ss.ss,A),A,nothing)
end

function block_symbolic_setup(block::TriformBlock,solver::LinearSolver,mat::AbstractMatrix,x::AbstractVector)
  y  = restrict_blocks(x,block.ids)
  uh = FEFunction(block.param,y)
  f(u,v) = block.f(uh,u,v)
  A = assemble_matrix(f,block.assem,block.trial,block.test)
  return BlockSS(block,symbolic_setup(solver,A,y),A)
end
function block_numerical_setup(block::TriformBlock,ss::BlockSS,mat::AbstractMatrix,x::AbstractVector)
  A = ss.cache
  return BlockNS(block,numerical_setup(ss.ss,A,x),A,nothing)
end
function block_numerical_setup!(block::TriformBlock,ns::BlockNS,mat::AbstractMatrix,x::AbstractVector)
  y = restrict_blocks(x,block.ids)
  uh = FEFunction(block.param,y)
  f(u,v) = block.f(uh,u,v)
  A = assemble_matrix!(f,ns.mat,block.assem,block.trial,block.test)
  numerical_setup!(ns.ns,A,y)
  return ns
end

function block_offdiagonal_setup(block::BiformBlock,mat::AbstractMatrix)
  A = assemble_matrix(block.f,block.assem,block.trial,block.test)
  return A
end
function block_offdiagonal_setup(block::TriformBlock,mat::AbstractMatrix,x::AbstractVector)
  y  = restrict_blocks(x,block.ids)
  uh = FEFunction(block.param,y)
  f(u,v) = block.f(uh,u,v)
  A = assemble_matrix(f,block.assem,block.trial,block.test)
  return A
end
function block_offdiagonal_setup!(cache,block::TriformBlock,mat::AbstractMatrix,x::AbstractVector)
  y  = restrict_blocks(x,block.ids)
  uh = FEFunction(block.param,y)
  f(u,v) = block.f(uh,u,v)
  A = assemble_matrix!(f,cache,block.assem,block.trial,block.test)
  return A
end

# CompositeBlock, i.e something that takes from the system matrix and adds another contribution to it.
# How do we deal with different sparsity patterns? Not trivial...
