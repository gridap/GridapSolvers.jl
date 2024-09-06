using Gridap
using Gridap.Geometry, Gridap.Arrays

using GridapSolvers
using GridapSolvers: PatchBasedSmoothers  

function generate_dual_graph(
  topo::GridTopology{Dc}, D::Integer = Dc
) where Dc
  @assert 0 < D <= Dc
  edge_to_face = Geometry.get_faces(topo,D-1,D)
  n_faces = Geometry.num_faces(topo,D)
  return generate_dual_graph(edge_to_face,n_faces)
end

# Given a table `edge_to_face`, creates the dual graph `face_to_face`.
function generate_dual_graph(
  edge_to_face::Table,
  n_faces = maximum(edge_to_face.data)
)
  n_edges = length(edge_to_face)

  ptrs = zeros(Int,n_faces+1)
  for e in 1:n_edges
    faces = view(edge_to_face,e)
    if length(faces) > 1
      @assert length(faces) == 2
      f1, f2 = faces
      ptrs[f1+1] += 1
      ptrs[f2+1] += 1
    end
  end
  Arrays.length_to_ptrs!(ptrs)

  data = zeros(Int,ptrs[end]-1)
  for e in 1:n_edges
    faces = view(edge_to_face,e)
    if length(faces) > 1
      f1, f2 = faces
      data[ptrs[f1]] = f2
      data[ptrs[f2]] = f1
      ptrs[f1] += 1
      ptrs[f2] += 1
    end
  end
  Arrays.rewind_ptrs!(ptrs)

  return Table(data,ptrs)
end

model = CartesianDiscreteModel((0,1,0,1),(2,2))
topo = get_grid_topology(model)

PD = PatchDecomposition(model;patch_boundary_style=PatchBasedSmoothers.PatchBoundaryInclude())

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
Vh = FESpace(model,reffe)
Ph = PatchFESpace(Vh,PD,reffe)

Ω  = Triangulation(model)
Ωp = Triangulation(PD)
Ωc = Closure(PD)

dΩ  = Measure(Ω,2*order)
dΩp = Measure(Ωp,2*order)
dΩc = Measure(Ωc,2*order)

biform(u,v,dΩ) = ∫(u⋅v)dΩ
a(u,v)  = biform(u,v,dΩ)
ap(u,v) = biform(u,v,dΩp)
ac(u,v) = biform(u,v,dΩc)

A  = assemble_matrix(a,Vh,Vh)
Ap = assemble_matrix(ap,Ph,Ph)
Ac = assemble_matrix(ac,Ph,Ph)

vanka_PD = PatchDecomposition(model;patch_boundary_style=PatchBasedSmoothers.PatchBoundaryExclude())
vanka = PatchBasedSmoothers.VankaSolver(Vh,vanka_PD)
vanka_ids = vanka.patch_ids

dof_to_pdof = Ph.dof_to_pdof
patch_cell_ids = get_cell_dof_ids(Ph)
pdof_to_dof = flatten_partition(dof_to_pdof)

for (patch,ids) in enumerate(vanka_ids)
  patch_ids = unique(vcat(PatchBasedSmoothers.patch_view(PD,patch_cell_ids,patch)...))

  perm = sortperm(pdof_to_dof[patch_ids])
  patch_ids = patch_ids[perm]

  println("> Patch $patch")
  println("   > Vanka: ",ids)
  println("   > Space: ",pdof_to_dof[patch_ids])
  @assert ids == pdof_to_dof[patch_ids]

  A_vanka = A[ids,ids]
  A_vanka_bis = Ac[patch_ids,patch_ids]
  @assert A_vanka ≈ A_vanka_bis
end

b(u,v) = ∫(u⋅v)dΩc + ∫(u⋅v)dΩp
B = assemble_matrix(b,Ph,Ph)
@assert B ≈ Ac + Ap

# Multifield

Xh = MultiFieldFESpace([Vh,Vh])
Zh = MultiFieldFESpace([Ph,Ph])

biform_mf((u1,u2),(v1,v2),dΩ) = ∫(u1⋅v1 + u2⋅v2)dΩ
a_mf(u,v)  = biform_mf(u,v,dΩ)
ap_mf(u,v) = biform_mf(u,v,dΩp)
ac_mf(u,v) = biform_mf(u,v,dΩc)

A  = assemble_matrix(a_mf,Xh,Xh)
Ap = assemble_matrix(ap_mf,Zh,Zh)
Ac = assemble_matrix(ac_mf,Zh,Zh)
