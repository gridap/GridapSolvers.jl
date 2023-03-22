module BlockDiagonalSmoothersTests

using Gridap
using Gridap.MultiField
using BlockArrays
using LinearAlgebra
using FillArrays

using GridapSolvers

u(x) = VectorValue(x[1],x[2])
f(x) = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))

p(x) = x[1] + x[2]
g(x) = -Δ(p)(x)

D = 2
n = 10
domain = Tuple(repeat([0,1],D))
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

order  = 1
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["boundary"])

reffeₚ = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags=["boundary"])

U = TrialFESpace(V,u)
P = TrialFESpace(Q,p)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

degree = 2*order + 1
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

a((u,p),(v,q)) = ∫( v⊙u + q⋅p)dΩ

A,b = AffineFEOperator(a,l,X,Y)

dof_ids = get_free_dof_ids(X)
ranges  = map(i->dof_ids[Block(i)],1:blocklength(dof_ids))
solvers = Fill(BackslashSolver(),2)

P = BlockDiagonalPreconditioner(A,ranges,solvers)
Pss = symbolic_setup(P,A)
Pns = numerical_setup(Pss,A)

x = get_free_dof_values(zero(X))
ldiv!(x,Pns,b)


end