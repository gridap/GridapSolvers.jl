
using LinearAlgebra
using Test
using PartitionedArrays
using Gridap
using Gridap.Arrays
using Gridap.Helpers
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.FESpaces
using GridapDistributed
using FillArrays

using GridapSolvers
import GridapSolvers.PatchBasedSmoothers as PBS

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
Λ  = Skeleton(model)
dΛ = Measure(Λ,3)
Γ  = Boundary(model)
dΓ = Measure(Γ,3)

aΩ(u,v) = ∫(v⋅u)*dΩ
aΛ(u,v) = ∫(jump(v)⋅jump(u))*dΛ
aΓ(u,v) = ∫(v⋅u)*dΓ
a(u,v)  = aΩ(u,v) + aΛ(u,v) + aΓ(u,v)
l(v)    = ∫(1*v)*dΩ

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

############################################################################################
# Integration 

Dc = 2
Df       = Dc -1 
model    = PD.model
labeling = get_face_labeling(model)

u = get_trial_fe_basis(Vh)
v = get_fe_basis(Vh)

patch_cells = PD.patch_cells

# Boundary
is_boundary = get_face_mask(labeling,["boundary"],Df)
patch_faces = PBS.get_patch_faces(PD,1,is_boundary)
pfaces_to_pcells = PBS.get_pfaces_to_pcells(PD,Df,patch_faces)

glue = get_glue(Γ,Val(Df))
mface_to_tface = Gridap.Arrays.find_inverse_index_map(glue.tface_to_mface,num_faces(model,Df))
patch_faces_data = lazy_map(Reindex(mface_to_tface),patch_faces.data)

contr = aΓ(u,v)
vecdata = first(contr.dict)[2]
patch_vecdata = lazy_map(Reindex(vecdata),patch_faces_data)

cell_dof_ids = get_cell_dof_ids(Ph)
face_dof_ids = lazy_map(Reindex(cell_dof_ids),lazy_map(x->x[1],pfaces_to_pcells))

res = ([patch_vecdata],[face_dof_ids],[face_dof_ids])
assemble_matrix(assembler_P,res)

# Interior
is_interior = get_face_mask(labeling,["interior"],Df)
patch_faces = PBS.get_patch_faces(PD,Df,is_interior)
pfaces_to_pcells = PBS.get_pfaces_to_pcells(PD,Df,patch_faces)


############################################################################################

β = 10
aΩ(u,v) = ∫(v⋅u)*dΩₚ
aΓ(u,v) = ∫(β⋅jump(v)⋅jump(u))*dΛₚ

ap(u,v) = aΩ(u,v) + aΓ(u,v)

assembler_P = SparseMatrixAssembler(Ph,Ph)

v = get_fe_basis(Ph)
u = get_trial_fe_basis(Ph)
contr = ap(u,v)

cellmat,rows,cols = collect_cell_matrix(Ph,Ph,contr)


Ahp = assemble_matrix(ap,assembler_P,Ph,Ph)
