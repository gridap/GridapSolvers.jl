
using Gridap
using Gridap.Adaptivity, Gridap.Geometry, Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Arrays
using Gridap.MultiField, Gridap.Algebra

using GridapSolvers
using GridapSolvers.BlockSolvers, GridapSolvers.LinearSolvers

using FillArrays

# Parameters

Dc = 3
reftype = 1

u_sol(x) = (Dc == 2) ? VectorValue(x[1]^2*x[2], -x[1]*x[2]^2) : VectorValue(x[1]^2*x[2], -x[1]*x[2]^2,0.0)
p_sol(x) = (x[1] - 1.0/2.0)

domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
nc = (Dc == 2) ? (20,20) : (4,4,4)
model = simplexify(CartesianDiscreteModel(domain,nc))

min_order = (reftype == 1) ? Dc : Dc-1
order = max(2,min_order)
poly  = (Dc == 2) ? TRI : TET
rrule = (reftype == 1) ? Adaptivity.BarycentricRefinementRule(poly) : Adaptivity.PowellSabinRefinementRule(poly)
reffes = Fill(LagrangianRefFE(VectorValue{Dc,Float64},poly,order),Adaptivity.num_subcells(rrule))
reffe_u = Adaptivity.MacroReferenceFE(rrule,reffes)
reffe_p = LagrangianRefFE(Float64,poly,order-1)

qdegree = 2*order
quad  = Quadrature(poly,Adaptivity.CompositeQuadrature(),rrule,qdegree)

V = FESpace(model,reffe_u,dirichlet_tags=["boundary"])
Q = FESpace(model,reffe_p,conformity=:L2,constraint=:zeromean)
U = TrialFESpace(V,u_sol)

Ω = Triangulation(model)
dΩ = Measure(Ω,quad)
@assert abs(sum(∫(p_sol)dΩ)) < 1.e-15
@assert abs(sum(∫(divergence(u_sol))dΩ)) < 1.e-15

mfs = BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

α = 1.e2
f(x) = -Δ(u_sol)(x) + ∇(p_sol)(x)
graddiv(u,v)  = ∫(α*(∇⋅v)⋅(∇⋅u))dΩ
lap(u,v) = ∫(∇(u)⊙∇(v))dΩ
a((u,p),(v,q)) = lap(u,v) + ∫(divergence(u)*q - divergence(v)*p)dΩ + graddiv(u,v)
l((v,q)) = ∫(f⋅v)dΩ

op = AffineFEOperator(a,l,X,Y)
A = get_matrix(op)
b = get_vector(op)

solver_u = LUSolver()
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)
solver_p.log.depth = 2

bblocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)]
coeffs = [1.0 1.0;
          0.0 1.0]  
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-10,rtol=1.e-12,verbose=true)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)
xh = FEFunction(X,x)

uh, ph = xh
eh_u = uh - u_sol
eh_p = ph - p_sol
sum(∫(eh_u⋅eh_u)dΩ)
sum(∫(eh_p*eh_p)dΩ)

norm(assemble_vector( q -> ∫(divergence(uh)*q)dΩ, Q))
abs(sum(∫(divergence(uh))dΩ))
