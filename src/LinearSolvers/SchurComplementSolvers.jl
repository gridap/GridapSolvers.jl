
"""
  Schur complement solver

  [A B] ^ -1    [Ip  -A^-1 B]   [A^-1     ]   [   Ip       ]
  [C D]       = [       Iq  ] ⋅ [     S^-1] ⋅ [-C A^-1   Iq]

  where S = D - C A^-1 B
"""
struct SchurComplementSolver{T1,T2,T3,T4} <: Gridap.Algebra.LinearSolver
  A :: T1
  B :: T2
  C :: T3 
  S :: T4
  function SchurComplementSolver(A::Gridap.Algebra.NumericalSetup,
                                 B::AbstractMatrix,
                                 C::AbstractMatrix,
                                 S::Gridap.Algebra.NumericalSetup)
    T1 = typeof(A)
    T2 = typeof(B)
    T3 = typeof(C)
    T4 = typeof(S)
    return new{T1,T2,T3,T4}(A,B,C,S)
  end
end

struct SchurComplementSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(s::SchurComplementSolver,A::AbstractMatrix) 
  SchurComplementSymbolicSetup(s)
end

struct SchurComplementNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
  mat
  ranges
  caches
end 

function get_shur_complement_caches(B::AbstractMatrix,C::AbstractMatrix)
  du1   = allocate_col_vector(C)
  du2   = allocate_col_vector(C)
  dp    = allocate_col_vector(B)

  rv_u  = allocate_row_vector(B)
  rv_p  = allocate_row_vector(C)
  return (du1,du2,dp,rv_u,rv_p)
end

function get_block_ranges(B::AbstractMatrix,C::AbstractMatrix)
  u_range = 1:size(C,2)
  p_range = size(C,2) .+ (1:size(B,2))
  return u_range, p_range
end

function get_block_ranges(B::PSparseMatrix,C::PSparseMatrix)
  ranges = map(own_values(B),own_values(C)) do B,C
    get_block_ranges(B,C)
  end
  return ranges
end

function Gridap.Algebra.numerical_setup(ss::SchurComplementSymbolicSetup,mat::AbstractMatrix)
  s   = ss.solver
  B,C = s.B, s.C
  ranges = compute_block_ranges(C,B)
  caches = get_shur_complement_caches(B,C)
  return SchurComplementNumericalSetup(s,mat,ranges,caches)
end

function to_blocks!(x::AbstractVector,u,p,ranges)
  u_range, p_range = ranges
  u .= x[u_range]
  p .= x[p_range]
  return u,p
end

function to_blocks!(x::PVector,u,p,ranges)
  map(own_values(x),own_values(u),own_values(p),ranges) do x,u,p,ranges
    to_blocks!(x,u,p,ranges)
  end
  consistent!(u) |> fetch
  consistent!(p) |> fetch
  return u,p
end

function to_global!(x::AbstractVector,u,p,ranges)
  u_range, p_range = ranges
  x[u_range] .= u
  x[p_range] .= p
  return x
end

function to_global!(x::PVector,u,p,ranges)
  map(own_values(x),own_values(u),own_values(p),ranges) do x,u,p,ranges
    to_global!(x,u,p,ranges)
  end
  consistent!(x) |> fetch
  return x
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::SchurComplementNumericalSetup,y::AbstractVector)
  s = ns.solver
  A,B,C,S = s.A,s.B,s.C,s.S
  du1,du2,dp,rv_u,rv_p = ns.caches

  # Split y into blocks
  to_blocks!(y,rv_u,rv_p,ns.ranges)

  # Solve Schur complement
  solve!(du1,A,rv_u)        # du1 = A^-1 y_u
  mul!(rv_p,C,du1,1.0,-1.0) # b1  = C*du1 - y_p
  solve!(dp,S,rv_p)         # dp  = S^-1 b1
  mul!(rv_u,B,dp)           # b2  = B*dp
  solve!(du2,A,rv_u)        # du2 = A^-1 b2
  du1 .-= du2               # du  = du1 - du2

  # Assemble into global
  to_global!(x,du1,dp,ns.ranges)

  return x
end
