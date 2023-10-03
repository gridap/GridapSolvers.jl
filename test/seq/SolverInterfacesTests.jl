
using Gridap

using GridapSolvers
import GridapSolvers.SolverInterfaces as SI
using GridapSolvers.LinearSolvers

sol(x) = x[1] + x[2]
f(x)   = -Δ(sol)(x)

mesh_partition = (10,10)
domain = (0,1,0,1)
model  = CartesianDiscreteModel(domain,mesh_partition)

order  = 1
qorder = order*2 + 1
reffe  = ReferenceFE(lagrangian,Float64,order)
Vh     = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
Uh     = TrialFESpace(Vh,sol)
u      = interpolate(sol,Uh)

Ω      = Triangulation(model)
dΩ     = Measure(Ω,qorder)
a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
l(v)   = ∫(v⋅f)*dΩ
op = AffineFEOperator(a,l,Uh,Vh)
A, b = get_matrix(op), get_vector(op);
P = JacobiLinearSolver()

solver = LinearSolvers.CGSolver(P;rtol=1.e-8,verbose=true)
ns = numerical_setup(symbolic_setup(solver,A),A)
x = LinearSolvers.allocate_col_vector(A)
solve!(x,ns,b)

solver = LinearSolvers.GMRESSolver(10;Pl=P,rtol=1.e-8,verbose=2)
ns = numerical_setup(symbolic_setup(solver,A),A)
x = LinearSolvers.allocate_col_vector(A)
solve!(x,ns,b)
