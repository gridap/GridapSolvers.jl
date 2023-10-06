
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

struct SchurComplementNumericalSetup{A,B,C} <: Gridap.Algebra.NumericalSetup
  solver::A
  mat   ::B
  caches::C
end 

function get_shur_complement_caches(B::AbstractMatrix,C::AbstractMatrix)
  du = allocate_col_vector(C)
  bu = allocate_col_vector(C)
  bp = allocate_col_vector(B)
  return du,bu,bp
end

function Gridap.Algebra.numerical_setup(ss::SchurComplementSymbolicSetup,mat::AbstractMatrix)
  s   = ss.solver
  caches = get_shur_complement_caches(s.B,s.C)
  return SchurComplementNumericalSetup(s,mat,caches)
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::SchurComplementNumericalSetup,y::AbstractBlockVector)
  s = ns.solver
  A,B,C,S = s.A,s.B,s.C,s.S
  du,bu,bp = ns.caches

  @check blocklength(x) == blocklength(y) == 2
  y_u = y[Block(1)]; y_p = y[Block(2)]
  x_u = x[Block(1)]; x_p = x[Block(2)]

  # Solve Schur complement
  solve!(x_u,A,y_u)                      # x_u = A^-1 y_u
  copy!(bp,y_p); mul!(bp,C,du,-1.0,1.0)  # bp  = C*(A^-1 y_u) - y_p
  solve!(x_p,S,bp)                       # x_p = S^-1 bp

  mul!(bu,B,x_p)         # bu  = B*x_p
  solve!(du,A,bu)        # du = A^-1 bu
  x_u .-= du             # x_u  = x_u - du

  return x
end
