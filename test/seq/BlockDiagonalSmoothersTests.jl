module BlockDiagonalSmoothersTests

using Gridap
using Gridap.MultiField
using BlockArrays
using LinearAlgebra
using FillArrays
using IterativeSolvers

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

order  = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["boundary"])

reffeₚ = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags=["boundary"])

U = TrialFESpace(V,u)
P = TrialFESpace(Q,p)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

degree = 2*(order + 1)
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)


# Global problem
a((u,p),(v,q)) = ∫( v⊙u + ∇(v)⊙∇(u) + q⋅p + ∇(q)⊙∇(p))dΩ
l((v,q)) = ∫( v⋅f + q⋅g)dΩ

op = AffineFEOperator(a,l,X,Y)
A,b = get_matrix(op), get_vector(op)
xh_star = solve(op)
x_star = get_free_dof_values(xh_star)

dof_ids = get_free_dof_ids(X)
ranges  = map(i->dof_ids[Block(i)],1:blocklength(dof_ids))
solvers = Fill(BackslashSolver(),2)

# Build using the global matrix
BDS   = BlockDiagonalSmoother(A,ranges,solvers)
BDSss = symbolic_setup(BDS,A)
BDSns = numerical_setup(BDSss,A)

x = get_free_dof_values(zero(X))
x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)

norm(x-x_star)

# Build using local weakforms
a1(u,v) = ∫(v⊙u + ∇(v)⊙∇(u))dΩ
a2(p,q) = ∫(q⋅p + ∇(q)⊙∇(p))dΩ
biforms = [a1,a2]

BDS   = BlockDiagonalSmoother(biforms,X,Y,solvers)
BDSss = symbolic_setup(BDS,A)
BDSns = numerical_setup(BDSss,A)

x = get_free_dof_values(zero(X))
x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)

norm(x-x_star)

end