
# LinearSolvers that depend on the non-linear solution
"""
function Gridap.Algebra.numerical_setup!(ns::Gridap.Algebra.LinearSolver,A::AbstractMatrix,x::AbstractVector)
  numerical_setup!(ns,A)
end

function allocate_solver_caches(ns::Gridap.Algebra.LinearSolver,args...;kwargs...)
  @abstractmethod
end
"""