
const NothingOrSolver = Union{Nothing,Gridap.Algebra.LinearSolver}

"""
  Computes the Krylov matrix-vector product 
  
  `y = Pl⁻¹⋅A⋅Pr⁻¹⋅x`

  by solving

  ```
    Pr⋅wr = x
    wl = A⋅wr
    Pl⋅y = wl
  ```
"""
function krylov_mul!(y,A,x,Pr,Pl,wr,wl)
  solve!(wr,Pr,x)
  mul!(wl,A,wr)
  solve!(y,Pl,wl)
end
function krylov_mul!(y,A,x,Pr,Pl::Nothing,wr,wl)
  solve!(wr,Pr,x)
  mul!(y,A,wr)
end
function krylov_mul!(y,A,x,Pr::Nothing,Pl,wr,wl)
  mul!(wl,A,x)
  solve!(y,Pl,wl)
end
function krylov_mul!(y,A,x,Pr::Nothing,Pl::Nothing,wr,wl)
  mul!(y,A,x)
end

"""
  Computes the Krylov residual 

  `r = Pl⁻¹(A⋅x - b)`

  by solving

  ```
    w = b - A⋅x
    Pl⋅r = w
  ```
"""
function krylov_residual!(r,x,A,b,Pl,w)
  mul!(w,A,x)
  w .= b .- w
  solve!(r,Pl,w)
end
function krylov_residual!(r,x,A,b,Pl::Nothing,w)
  mul!(r,A,x)
  r .= b .- r
end

# Lanczos algorithm

struct LanczosDiagnostic{T}
  k::Base.RefValue{Int}
  delta::Vector{T}
  gamma::Vector{T}
  function LanczosDiagnostic(delta::Vector{T},gamma::Vector{T}) where T
    new{T}(Ref(0),delta,gamma)
  end
end

LanczosDiagnostic(max_iters::Int, T::Type = Float64) = LanczosDiagnostic(zeros(T,max_iters),zeros(T,max_iters))

function SolverInterfaces.reset!(x::LanczosDiagnostic)
  x.k[] = 0
  fill!(x.delta, 0)
  fill!(x.gamma, 0)
  return x
end

function SolverInterfaces.update!(x::LanczosDiagnostic, δ::Real, γ::Real)
  x.k[] += 1
  x.delta[x.k[]] = δ
  x.gamma[x.k[]] = γ
  return x
end

function estimate!(x::LanczosDiagnostic)
  k = x.k[]
  (k < 2) && return 1.0
  M = SymTridiagonal(view(x.delta,1:k), view(x.gamma,2:k))
  λ = eigvals(M)
  λmin, λmax = extrema(λ)
  return abs(λmax / λmin)
end
