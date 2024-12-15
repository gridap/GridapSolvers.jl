
# LinearSolvers that depend on the non-linear solution

function Gridap.Algebra.symbolic_setup(ns::Gridap.Algebra.LinearSolver,A::AbstractMatrix,x::AbstractVector)
  symbolic_setup(ns,A)
end

function Gridap.Algebra.numerical_setup(ns::Gridap.Algebra.SymbolicSetup,A::AbstractMatrix,x::AbstractVector)
  numerical_setup(ns,A)
end

function Gridap.Algebra.numerical_setup!(ns::Gridap.Algebra.NumericalSetup,A::AbstractMatrix,x::AbstractVector)
  numerical_setup!(ns,A)
end

# Similar to PartitionedArrays.matching_local_indices, but cheaper since 
# we do not try to match the indices.

function same_local_indices(a::PRange,b::PRange)
  partition(a) === partition(b) && return true
  c = map(===,partition(a),partition(b))
  reduce(&,c,init=true)
end

function same_local_indices(a::BlockPRange,b::BlockPRange)
  c = map(same_local_indices,blocks(a),blocks(b))
  reduce(&,c,init=true)
end

# The following is needed, otherwise the input vector `x` potentially does not match 
# the domain of the operator `op`. 

function Algebra.solve!(x::PVector,ls::LinearSolver,op::AffineOperator,cache::Nothing)
  A = op.matrix
  b = op.vector
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  y = allocate_in_domain(A)
  copy!(y,x)
  solve!(y,ns,b)
  copy!(x,y)
  consistent!(x) |> wait
  return ns, y
end

function Algebra.solve!(x::PVector,ls::LinearSolver,op::AffineOperator,cache,newmatrix::Bool)
  A = op.matrix
  b = op.vector
  ns, y = cache
  if newmatrix
    numerical_setup!(ns,A)
  end
  copy!(y,x)
  solve!(y,ns,b)
  copy!(x,y)
  consistent!(x) |> wait
  return cache
end
