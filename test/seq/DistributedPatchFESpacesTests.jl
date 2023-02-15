module DistributedPatchFESpacesTests

using Test
using Gridap
using GridapDistributed
using PartitionedArrays
using FillArrays

include("../../src/PatchBasedSmoothers/PatchBasedSmoothers.jl")
import .PatchBasedSmoothers as PBS

backend = SequentialBackend()
ranks = (1,2)
parts = get_part_ids(backend,ranks)

domain = (0.0,1.0,0.0,1.0)
partition = (2,4)
model = CartesianDiscreteModel(parts,domain,partition)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vh = TestFESpace(model,reffe)
PD = PBS.PatchDecomposition(model,patch_boundary_style=PBS.PatchBoundaryInclude())
Ph = PBS.PatchFESpace(model,reffe,Gridap.ReferenceFEs.H1Conformity(),PD,Vh)

w, w_sums = PBS.compute_weight_operators(Ph,Vh);

xP = PVector(1.0,Ph.gids)
yP = PVector(0.0,Ph.gids)
x = PVector(1.0,Vh.gids)
y = PVector(0.0,Vh.gids)

PBS.prolongate!(yP,Ph,x)
PBS.inject!(y,Ph,yP,w,w_sums)


assembler = SparseMatrixAssembler(Ph,Ph)
Ωₚ  = Triangulation(PD)
dΩₚ = Measure(Ωₚ,2*order+1)
a(u,v) = ∫(∇(v)⋅∇(u))*dΩₚ
l(v) = ∫(1*v)*dΩₚ

Ah = assemble_matrix(a,assembler,Ph,Ph)
fh = assemble_vector(l,assembler,Ph)



end