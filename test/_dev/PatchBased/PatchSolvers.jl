using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.MultiField

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.LinearSolvers
using GridapSolvers.PatchBasedSmoothers

using LinearAlgebra

function mean_value(p,dΩ)
  sum(∫(p)dΩ)
end

model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(4,4)))

order = 2
qdegree = 2*order+1
reffe_u = LagrangianRefFE(VectorValue{2,Float64},QUAD,order)
reffe_p = LagrangianRefFE(Float64,QUAD,order-1;space=:P)
reffe_λ = LagrangianRefFE(Float64,QUAD,0)

Vh = TestFESpace(model,reffe_u;dirichlet_tags="boundary")
Qh = TestFESpace(model,reffe_p,conformity=:L2)
Λh = ConstantFESpace(model)
Xh = MultiFieldFESpace([Vh,Qh,Λh])

qdegree = 2*order+1
f(x) = VectorValue(1.0,1.0)
Π = LocalProjectionMap(divergence,Qh,qdegree)
a(u,v,dΩ) = ∫(∇(u)⊙∇(v))dΩ
b((u,p),(v,q),dΩ) = ∫(p*(∇⋅v) + q*(∇⋅u))dΩ
d(u,v,dΩ) = ∫(Π(v)*Π(u))dΩ
c((p,λ),(q,μ),dΩ) = ∫(λ*q + μ*p)dΩ

β = 1.0
biform((u,p,λ),(v,q,μ),dΩ) = a(u,v,dΩ) + b((u,p),(v,q),dΩ) + β*c((p,λ),(q,μ),dΩ) #+ d(u,v,dΩ)
liform((v,q),dΩ) = ∫(f⋅v)dΩ

PD = PatchDecomposition(model,patch_boundary_style=PatchBasedSmoothers.PatchBoundaryInclude())

Vp = PatchFESpace(Vh,PD,reffe_u)
Qp = PatchFESpace(Qh,PD,reffe_p)
Λp = PatchFESpace(Λh,PD,reffe_λ)
Zh = MultiFieldFESpace([Vp,Qp,Λp])

Ω = Triangulation(model)
dΩ = Measure(Ω,4)

Ωp = Triangulation(PD)
dΩp = Measure(Ωp,4)

ap(u,v) = a(u,v,dΩp)
ps = PatchBasedSmoothers.PatchSolver(Vh,PD,ap,reffe_u)

patch_ids = ps.patch_ids
patch_cell_lids = ps.patch_cell_lids

A = assemble_matrix((u,v)->a(u,v,dΩ),Vh,Vh)
ss = symbolic_setup(ps,A)
ns = numerical_setup(ss,A)

rhs = assemble_vector(v -> ∫(f⋅v)dΩ,Vh)
x = zeros(num_free_dofs(Vh))
solve!(x,ns,rhs)

biformp((u,p,λ),(v,q,μ)) = biform((u,p,λ),(v,q,μ),dΩp)
ps_mf = PatchBasedSmoothers.PatchSolver(Xh,PD,biformp,[reffe_u,reffe_p,reffe_λ])

patch_ids = ps_mf.patch_ids
patch_cell_lids = ps_mf.patch_cell_lids

A = assemble_matrix((u,v)->biform(u,v,dΩ),Xh,Xh)
ss = symbolic_setup(ps_mf,A)
ns = numerical_setup(ss,A)

rhs = assemble_vector(y -> liform(y,dΩ),Xh)
x = zeros(num_free_dofs(Xh))
solve!(x,ns,rhs)

