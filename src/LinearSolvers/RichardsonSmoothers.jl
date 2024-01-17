"""
    struct RichardsonSmoother <: LinearSolver
      ...
    end

    RichardsonSmoother(M::LinearSolver,niter::Int=1,ω::Float64=1.0)

  Performs `niter` Richardson iterations with relaxation parameter `ω` 
  using the linear solver `M`.
  
  Updates both the solution `x` and the residual `r` in place.
"""
struct RichardsonSmoother{A} <: Gridap.Algebra.LinearSolver
  M     :: A
  niter :: Int64
  ω     :: Float64
  function RichardsonSmoother(
    M::Gridap.Algebra.LinearSolver,
    niter::Integer=1,
    ω::Real=1.0
  )
    A = typeof(M)
    return RichardsonSmoother{A}(M,niter,ω)
  end
end

struct RichardsonSmootherSymbolicSetup{A,B} <: Gridap.Algebra.SymbolicSetup
  smoother :: RichardsonSmoother{A}
  Mss      :: B
end

function Gridap.Algebra.symbolic_setup(smoother::RichardsonSmoother,mat::AbstractMatrix)
  Mss = symbolic_setup(smoother.M,mat)
  return RichardsonSmootherSymbolicSetup(smoother,Mss)
end

mutable struct RichardsonSmootherNumericalSetup{A,B,C,D,E} <: Gridap.Algebra.NumericalSetup
  smoother :: RichardsonSmoother{A}
  A        :: B
  Adx      :: C
  dx       :: D
  Mns      :: E
end

function Gridap.Algebra.numerical_setup(ss::RichardsonSmootherSymbolicSetup, A::AbstractMatrix)
  Adx = allocate_in_range(A)
  dx  = allocate_in_domain(A)
  Mns = numerical_setup(ss.Mss,A)
  return RichardsonSmootherNumericalSetup(ss.smoother,A,Adx,dx,Mns)
end

function Gridap.Algebra.numerical_setup!(ns::RichardsonSmootherNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Mns,A)
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::RichardsonSmootherNumericalSetup,r::AbstractVector)
  Adx,dx,Mns = ns.Adx,ns.dx,ns.Mns
  niter, ω = ns.smoother.niter, ns.smoother.ω

  iter = 1
  while iter <= niter
    solve!(dx,Mns,r)
    dx .= ω .* dx
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
