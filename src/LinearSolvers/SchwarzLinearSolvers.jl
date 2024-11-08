
# TODO: 
# - Implement the multiplicative case
# - Add support for weights/averaging when aggregating the additive case?

struct SchwarzLinearSolver{T,S,A} <: Algebra.LinearSolver
  local_solvers::A
  function SchwarzLinearSolver(
    solver::Union{S,AbstractVector{<:S}};
    type = :additive
  ) where S <: Algebra.LinearSolver
    @check type in (:additive,:multiplicative)
    @notimplementedif type == :multiplicative # TODO
    A = typeof(solver)
    new{type,S,A}(solver)
  end
end

struct SchwarzSymbolicSetup{T,S,A,B} <: Algebra.SymbolicSetup
  solver::SchwarzLinearSolver{T,S,A}
  local_ss::B
end

function Algebra.symbolic_setup(s::SchwarzLinearSolver,mat::AbstractMatrix)
  # TODO: This is where we should compute the comm coloring for the multiplicative case
  expand(s) = map(m -> s,partition(mat))
  expand(s::AbstractVector) = s

  local_solvers = expand(s.local_solvers)
  local_ss = map(symbolic_setup,local_solvers,partition(mat))
  return SchwarzSymbolicSetup(s,local_ss)
end

struct SchwarzNumericalSetup{T,S,A,B} <: Algebra.NumericalSetup
  solver::SchwarzLinearSolver{T,S,A}
  local_ns::B
end

function Algebra.numerical_setup(ss::SchwarzSymbolicSetup,mat::PSparseMatrix)
  local_ns = map(numerical_setup,ss.local_ss,partition(mat))
  return SchwarzNumericalSetup(ss.solver,local_ns)
end

function Algebra.solve!(x::PVector,ns::SchwarzNumericalSetup{:additive},b::PVector)
  map(solve!,partition(x),ns.local_ns,partition(b))
  assemble!(x) |> wait
  consistent!(x) |> wait
  return x
end
