
using Gridap
using GridapSolvers

using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Geometry
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.LinearSolvers

u_exact(x) = VectorValue(-x[1],x[2])
p_exact(x) = x[1] + x[2]

model = CartesianDiscreteModel((0,1,0,1),(2,2))
PD = PatchDecomposition(model)

Ω = Triangulation(model)
Ωp = Triangulation(PD)

order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

Vh = TestFESpace(model,reffe_u,dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u_exact)

Qh = TestFESpace(model,reffe_p,conformity=:L2,constraint=:zeromean)

topo = get_grid_topology(model)
#patches_mask = fill(false,PatchBasedSmoothers.num_patches(PD))
patches_mask = get_isboundary_face(topo,0)
Ph = PatchFESpace(Vh,PD,reffe_u;conformity=H1Conformity(),patches_mask=patches_mask)
Lh = PatchFESpace(Qh,PD,reffe_p;conformity=L2Conformity(),patches_mask=patches_mask)

Xh = MultiFieldFESpace([Uh,Qh])
Yh = MultiFieldFESpace([Vh,Qh])
Zh = MultiFieldFESpace([Ph,Lh])

qdegree = 2*order
dΩ = Measure(Ω,qdegree)
dΩp = Measure(Ωp,qdegree)

p_mean = sum(∫(p_exact)dΩ)/sum(∫(1)dΩ)
p_exact_zm(x) = p_exact(x) - p_mean
f(x) = Δ(u_exact)(x) + ∇(p_exact_zm)(x)
liform((v,q),dΩ) = ∫(v⋅f)dΩ
biform((u,p),(v,q),dΩ) = ∫(∇(u)⊙∇(v) - (∇⋅v)*p - (∇⋅u)*q)dΩ
a(x,y) = biform(x,y,dΩ)
l(y) = liform(y,dΩ)
ap(x,y) = biform(x,y,dΩp)

op = AffineFEOperator(a,l,Xh,Yh)
A = get_matrix(op) 
b = get_vector(op)
x_exact = A\b

P = PatchBasedSmoothers.PatchBasedLinearSolver(ap,Zh,Yh)
P_ns = numerical_setup(symbolic_setup(P,A),A)
x = zeros(num_free_dofs(Yh))
solve!(x,P_ns,b)

solver = FGMRESSolver(10,P;verbose=true)
#solver = GMRESSolver(10,verbose=true)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = zeros(num_free_dofs(Yh))
solve!(x,ns,b)

norm(A*x - b)

############################################################################################

patch_pcells = get_patch_cells_overlapped(PD)
pcell_to_pdofs_u = Ph.patch_cell_dofs_ids
pcell_to_pdofs_p = Lh.patch_cell_dofs_ids

patch_dofs_u = map(pcells -> unique(filter(x -> x > 0, vcat(pcell_to_pdofs_u[pcells]...))),patch_pcells)
patch_dofs_p = map(pcells -> unique(filter(x -> x > 0, vcat(pcell_to_pdofs_p[pcells]...))),patch_pcells)

o = num_free_dofs(Ph)
patch_dofs = map((du,dp) -> vcat(du,dp .+ o),patch_dofs_u,patch_dofs_p)

patch_cells = PD.patch_cells
patch_models = map(patch_cells) do cells
  pmodel  = DiscreteModelPortion(model,cells)
  pgrid   = UnstructuredGrid(get_grid(pmodel))
  ptopo   = get_grid_topology(pmodel)
  plabels = FaceLabeling(ptopo)
  UnstructuredDiscreteModel(pgrid,ptopo,plabels)
end

patch_spaces = map(patch_models) do pmodel
  Vhi = TestFESpace(pmodel,reffe_u;dirichlet_tags="boundary")
  Qhi = TestFESpace(pmodel,reffe_p,conformity=:L2,constraint=:zeromean)
  Yhi = MultiFieldFESpace([Vhi,Qhi])
end

Ap = assemble_matrix(ap,Zh,Zh)
patch_mats = map(dofs -> Ap[dofs,dofs],patch_dofs)
