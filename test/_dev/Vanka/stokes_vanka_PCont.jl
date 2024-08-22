
using Gridap
using GridapSolvers
using LinearAlgebra

using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Geometry
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.LinearSolvers

using GridapSolvers.PatchBasedSmoothers: PatchBoundaryExclude, PatchBoundaryInclude

function l2_norm(uh,dΩ)
  sqrt(sum(∫(uh⋅uh)dΩ))
end

function l2_error(uh,u_exact,dΩ)
  eh = uh - u_exact
  return sqrt(sum(∫(eh⋅eh)dΩ))
end

u_exact(x) = VectorValue(-x[1],x[2])
p_exact(x) = x[1] - 0.5

Dc = 2
model = CartesianDiscreteModel((0,1,0,1),(8,8))
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"newman",[5])
add_tag_from_tags!(labels,"dirichlet",[collect(1:4)...,6,7,8])

PD_e = PatchDecomposition(model,patch_boundary_style=PatchBoundaryExclude())
PD_i = PatchDecomposition(model,patch_boundary_style=PatchBoundaryInclude())

order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)

Vh = TestFESpace(model,reffe_u;dirichlet_tags="dirichlet")
Uh = TrialFESpace(Vh,u_exact)
Qh = TestFESpace(model,reffe_p)

PD = PD_i
Ph = PatchFESpace(Vh,PD_i,reffe_u;conformity=H1Conformity())
Lh = PatchFESpace(Qh,PD_e,reffe_p;conformity=H1Conformity())

Xh = MultiFieldFESpace([Uh,Qh])
Yh = MultiFieldFESpace([Vh,Qh])
Zh = MultiFieldFESpace([Ph,Lh])

qdegree = 2*order
Ω = Triangulation(model)
Ωp = Triangulation(PD)

dΩ = Measure(Ω,qdegree)
dΩp = Measure(Ωp,qdegree)

Γ = BoundaryTriangulation(model,tags="newman")
dΓ = Measure(Γ,qdegree)
n = get_normal_vector(Γ)

I_tensor = one(TensorValue{Dc,Dc,Float64})
f(x) = -Δ(u_exact)(x) - ∇(p_exact)(x)
σ(x) = ∇(u_exact)(x) + p_exact(x)*I_tensor
biform((u,p),(v,q),dΩ) = ∫(∇(u)⊙∇(v) + (∇⋅v)*p + (∇⋅u)*q)dΩ
a(x,y) = biform(x,y,dΩ)
l((v,q)) = ∫(v⋅f)dΩ + ∫(v⋅(σ⋅n) )dΓ

ap(x,y) = biform(x,y,dΩp)

op = AffineFEOperator(a,l,Xh,Yh)
A = get_matrix(op) 
b = get_vector(op)
cond(Matrix(A))
x_exact = A\b
uh_exact, ph_exact = FEFunction(Xh,x_exact)
l2_error(uh_exact,u_exact,dΩ)
l2_error(ph_exact,p_exact,dΩ)
l2_norm(∇⋅uh_exact,dΩ)

P = PatchBasedSmoothers.PatchBasedLinearSolver(ap,Zh,Yh)
P_ns = numerical_setup(symbolic_setup(P,A),A)
x = zeros(num_free_dofs(Yh))
solve!(x,P_ns,b)

solver = FGMRESSolver(10,P;rtol=1.e-8,verbose=true)
#solver = GMRESSolver(10,verbose=true)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = zeros(num_free_dofs(Yh))
solve!(x,ns,b)

norm(A*x - b)
