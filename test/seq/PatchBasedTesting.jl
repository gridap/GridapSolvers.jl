
using LinearAlgebra
using Test
using PartitionedArrays
using Gridap
using Gridap.Arrays
using Gridap.Helpers
using Gridap.Geometry
using Gridap.ReferenceFEs
using GridapDistributed
using FillArrays

using GridapSolvers
import GridapSolvers.PatchBasedSmoothers as PBS


function Gridap.Geometry.SkeletonTriangulation(PD::PatchDecomposition)
  model = PD.model
  labeling = get_face_labeling(model)
  topo = get_grid_topology(model)
  
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  
  c2e_map = get_faces(topo,2,1)
  patch_cells_edges = map(Reindex(c2e_map),patch_cells.data)
  
  is_boundary = get_face_mask(labeling,["boundary"],1)
  interior_edges = zeros(Int64,length(is_boundary))
  count = 1
  for i in 1:length(is_boundary)
    if !is_boundary[i] 
      interior_edges[i] = count
      count += 1
    end
  end
  
  edges_on_boundary = PD.patch_cells_faces_on_boundary[2]
  _patch_edges = map((E,mask)->E[.!mask],patch_cells_edges,edges_on_boundary)
  __patch_edges = map(E-> filter(e -> !is_boundary[e],E), _patch_edges)
  patch_edges = Gridap.Arrays.Table(__patch_edges)
  
  patch_edges_data = lazy_map(Reindex(interior_edges),patch_edges.data)
  
  Λ   = SkeletonTriangulation(model)
  return view(Λ,patch_edges_data)
end

backend = SequentialBackend()
ranks = (1,2)
parts = get_part_ids(backend,ranks)

domain = (0.0,1.0,0.0,1.0)
partition = (2,4)
model = CartesianDiscreteModel(domain,partition)

order = 1; reffe = ReferenceFE(lagrangian,Float64,order;space=:P); conformity = L2Conformity();
#order = 1; reffe = ReferenceFE(lagrangian,Float64,order); conformity = H1Conformity();
#order = 0; reffe = ReferenceFE(raviart_thomas,Float64,order); conformity = HDivConformity();
Vh = TestFESpace(model,reffe,conformity=conformity)
PD = PBS.PatchDecomposition(model)
Ph = PBS.PatchFESpace(model,reffe,conformity,PD,Vh)

# ---- Assemble systems ---- #

Ω  = Triangulation(model)
dΩ = Measure(Ω,2*order+1)
a(u,v) = ∫(v⋅u)*dΩ
l(v) = ∫(1*v)*dΩ

assembler = SparseMatrixAssembler(Vh,Vh)
Ah = assemble_matrix(a,assembler,Vh,Vh)
fh = assemble_vector(l,assembler,Vh)

sol_h = solve(LUSolver(),Ah,fh)

Ωₚ  = Triangulation(PD)
dΩₚ = Measure(Ωₚ,2*order+1)
ap(u,v) = ∫(v⋅u)*dΩₚ
lp(v) = ∫(1*v)*dΩₚ

assembler_P = SparseMatrixAssembler(Ph,Ph)
Ahp = assemble_matrix(ap,assembler_P,Ph,Ph)
fhp = assemble_vector(lp,assembler_P,Ph)

# Skeleton Triangulation
labeling = get_face_labeling(model)
topo = get_grid_topology(model)

patch_cells = Gridap.Arrays.Table(PD.patch_cells)

c2e_map = get_faces(topo,2,1)
patch_cells_edges = map(Reindex(c2e_map),patch_cells.data)

is_boundary = get_face_mask(labeling,["boundary"],1)
interior_edges = zeros(Int64,length(is_boundary))
count = 1
for i in 1:length(is_boundary)
  if !is_boundary[i] 
    interior_edges[i] = count
    count += 1
  end
end

edges_on_boundary = PD.patch_cells_faces_on_boundary[2]
_patch_edges = map((E,mask)->E[.!mask],patch_cells_edges,edges_on_boundary)
__patch_edges = map(E-> filter(e -> !is_boundary[e],E), _patch_edges)
patch_edges = Gridap.Arrays.Table(__patch_edges)

patch_edges_data = lazy_map(Reindex(interior_edges),patch_edges.data)

Λ   = SkeletonTriangulation(model)
Λₚ  = view(Λ,patch_edges_data)
dΛₚ = Measure(Λₚ,3)

β = 10
aΓ(u,v) = ∫(β⋅jump(v)⋅jump(u))*dΛₚ

v = get_fe_basis(Ph)
u = get_trial_fe_basis(Ph)
cf = (β⋅jump(v)⋅jump(u))
contr = aΓ(u,v)

matdata_edges = first(contr.dict)[2]

patch_edges_overlapped = Gridap.Arrays.Table(collect(1:length(patch_edges.data)),patch_edges.ptrs)
matdata_cells = lazy_map(Geometry.CombineContributionsMap(matdata_edges),patch_edges_overlapped)
