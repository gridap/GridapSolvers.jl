module DistributedPatchFESpacesTests

ENV["JULIA_MPI_BINARY"] = "system"
ENV["JULIA_MPI_PATH"] = "/usr/lib/x86_64-linux-gnu"

using LinearAlgebra
using Test
using PartitionedArrays
using Gridap
using Gridap.Helpers
using Gridap.Geometry
using GridapDistributed
using FillArrays

include("../../src/PatchBasedSmoothers/PatchBasedSmoothers.jl")
import .PatchBasedSmoothers as PBS

# This is needed for assembly
include("../../src/MultilevelTools/GridapFixes.jl")

include("../../src/LinearSolvers/RichardsonSmoothers.jl")

backend = SequentialBackend()
ranks = (1,2)
parts = get_part_ids(backend,ranks)

domain = (0.0,1.0,0.0,1.0)
partition = (2,4)
model = CartesianDiscreteModel(parts,domain,partition)

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vh = TestFESpace(model,reffe)
PD = PBS.PatchDecomposition(model)#,patch_boundary_style=PBS.PatchBoundaryInclude())
Ph = PBS.PatchFESpace(model,reffe,Gridap.ReferenceFEs.H1Conformity(),PD,Vh)

w, w_sums = PBS.compute_weight_operators(Ph,Vh);

xP = PVector(1.0,Ph.gids)
yP = PVector(0.0,Ph.gids)
x = PVector(1.0,Vh.gids)
y = PVector(0.0,Vh.gids)

PBS.prolongate!(yP,Ph,x)
PBS.inject!(y,Ph,yP,w,w_sums)
@test x ≈ y

PBS.inject!(x,Ph,xP,w,w_sums)
PBS.prolongate!(yP,Ph,x)
@test xP ≈ yP

Ωₚ  = Triangulation(PD)
dΩₚ = Measure(Ωₚ,2*order+1)
a(u,v) = ∫(∇(v)⋅∇(u))*dΩₚ
l(v) = ∫(1*v)*dΩₚ

assembler = SparseMatrixAssembler(Vh,Vh)
Ah = assemble_matrix(a,assembler,Vh,Vh)
fh = assemble_vector(l,assembler,Vh)

M = PBS.PatchBasedLinearSolver(a,Ph,Vh,LUSolver())
R = RichardsonSmoother(M,10,1.0/3.0)
Rss = symbolic_setup(R,Ah)
Rns = numerical_setup(Rss,Ah)

x = PBS._allocate_col_vector(Ah)
r = fh-Ah*x
exchange!(r)
solve!(x,Rns,r)

Mss = symbolic_setup(M,Ah)
Mns = numerical_setup(Mss,Ah)
solve!(x,Mns,r)

assembler_P = SparseMatrixAssembler(Ph,Ph)
Ahp = assemble_matrix(a,assembler_P,Ph,Ph)
fhp = assemble_vector(l,assembler_P,Ph)

lu = LUSolver()
luss = symbolic_setup(lu,Ahp)
luns = numerical_setup(luss,Ahp)

rp = PVector(0.0,Ph.gids)
PBS.prolongate!(rp,Ph,r)

rp_mat = PVector(0.0,Ahp.cols)
copy!(rp_mat,rp)
xp_mat = PVector(0.0,Ahp.cols)

solve!(xp_mat,luns,rp_mat)

xp = PVector(0.0,Ph.gids)
copy!(xp,xp_mat)

w, w_sums = PBS.compute_weight_operators(Ph,Vh);
PBS.inject!(x,Ph,xp,w,w_sums)

end