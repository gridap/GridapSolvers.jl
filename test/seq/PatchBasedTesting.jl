
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
Λₚ  = SkeletonTriangulation(PD)
dΛₚ = Measure(Λₚ,3)
Γₚ  = BoundaryTriangulation(PD)
dΓₚ = Measure(Γₚ,3)

aΩp(u,v) = ∫(v⋅u)*dΩₚ
aΛp(u,v) = ∫(jump(v)⋅jump(u))*dΛₚ
aΓp(u,v) = ∫(v⋅u)*dΓₚ
ap(u,v)  = aΩp(u,v) + aΛp(u,v) + aΓp(u,v)
lp(v)    = ∫(1*v)*dΩₚ

assembler_P = SparseMatrixAssembler(Ph,Ph)
Ahp = assemble_matrix(ap,assembler_P,Ph,Ph)
fhp = assemble_vector(lp,assembler_P,Ph)

