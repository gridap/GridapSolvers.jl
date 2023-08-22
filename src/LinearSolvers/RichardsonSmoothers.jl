

struct RichardsonSmoother{A,B} <: Gridap.Algebra.LinearSolver
  M                :: Gridap.Algebra.LinearSolver
  num_smooth_steps :: A
  damping_factor   :: B
end

function RichardsonSmoother(M::Gridap.Algebra.LinearSolver,
                            num_smooth_steps::Integer=1,
                            damping_factor::Real=1.0)
  A = typeof(num_smooth_steps)
  B = typeof(damping_factor)
  return RichardsonSmoother{A,B}(M,num_smooth_steps,damping_factor)
end

struct RichardsonSmootherSymbolicSetup{A} <: Gridap.Algebra.SymbolicSetup
  smoother :: RichardsonSmoother
  Mss      :: A
end

function Gridap.Algebra.symbolic_setup(smoother::RichardsonSmoother,mat::AbstractMatrix)
  Mss=symbolic_setup(smoother.M,mat)
  return RichardsonSmootherSymbolicSetup(smoother,Mss)
end

mutable struct RichardsonSmootherNumericalSetup{A,B,C,D} <: Gridap.Algebra.NumericalSetup
  smoother       :: RichardsonSmoother
  A              :: A
  Adx            :: B
  dx             :: C
  Mns            :: D
end

function Gridap.Algebra.numerical_setup(ss::RichardsonSmootherSymbolicSetup, A::AbstractMatrix{T}) where T
  Adx = zeros(size(A,1))
  dx  = zeros(size(A,2))
  Mns = numerical_setup(ss.Mss,A)
  return RichardsonSmootherNumericalSetup(ss.smoother,A,Adx,dx,Mns)
end

function Gridap.Algebra.numerical_setup(ss::RichardsonSmootherSymbolicSetup, A::PSparseMatrix)
  Adx = pfill(0.0,partition(axes(A,1)))
  dx  = pfill(0.0,partition(axes(A,2)))
  Mns = numerical_setup(ss.Mss,A)
  return RichardsonSmootherNumericalSetup(ss.smoother,A,Adx,dx,Mns)
end

function Gridap.Algebra.numerical_setup!(ns::RichardsonSmootherNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Mns,A)
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::RichardsonSmootherNumericalSetup,r::AbstractVector)
  Adx,dx,Mns = ns.Adx,ns.dx,ns.Mns

  iter = 1
  while iter <= ns.smoother.num_smooth_steps
      solve!(dx,Mns,r)
      dx .= ns.smoother.damping_factor .* dx
      x  .= x .+ dx
      mul!(Adx, ns.A, dx)
      r  .= r .- Adx
      iter += 1
  end
end

function LinearAlgebra.ldiv!(x::AbstractVector,ns::RichardsonSmootherNumericalSetup,b::AbstractVector)
  fill!(x,0.0)
  aux = copy(b)
  solve!(x,ns,aux)
  return x
end

