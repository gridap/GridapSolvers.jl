module DistributedPatchFESpacesTests

using LinearAlgebra
using Test
using PartitionedArrays
using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.ReferenceFEs
using GridapDistributed
using FillArrays

using GridapSolvers
import GridapSolvers.PatchBasedSmoothers as PBS

backend = SequentialBackend()
ranks = (1,2)
parts = get_part_ids(backend,ranks)

domain = (0.0,1.0,0.0,1.0)
partition = (2,4)
model = CartesianDiscreteModel(parts,domain,partition)

# order = 1
# reffe = ReferenceFE(lagrangian,Float64,order)
order = 0
reffe = ReferenceFE(raviart_thomas,Float64,order)
Vh = TestFESpace(model,reffe)
PD = PBS.PatchDecomposition(model)
Ph = PBS.PatchFESpace(model,reffe,DivConformity(),PD,Vh)

# ---- Testing Prolongation and Injection ---- #

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


# ---- Define solvers ---- #

LU   = LUSolver()
LUss = symbolic_setup(LU,Ahp)
LUns = numerical_setup(LUss,Ahp)

M   = PBS.PatchBasedLinearSolver(ap,Ph,Vh,LU)
Mss = symbolic_setup(M,Ah)
Mns = numerical_setup(Mss,Ah)

R   = RichardsonSmoother(M,10,1.0/3.0)
Rss = symbolic_setup(R,Ah)
Rns = numerical_setup(Rss,Ah)


# ---- Manual solve using LU ---- # 

x1_mat = PVector(0.5,Ah.cols)
r1_mat = fh-Ah*x1_mat
consistent!(r1_mat)

r1 = PVector(0.0,Vh.gids)
x1 = PVector(0.0,Vh.gids)
rp = PVector(0.0,Ph.gids)
xp = PVector(0.0,Ph.gids)
rp_mat = PVector(0.0,Ahp.cols)
xp_mat = PVector(0.0,Ahp.cols)

copy!(r1,r1_mat)
consistent!(r1)
PBS.prolongate!(rp,Ph,r1)

copy!(rp_mat,rp)
solve!(xp_mat,LUns,rp_mat)
copy!(xp,xp_mat)

w, w_sums = PBS.compute_weight_operators(Ph,Vh);
PBS.inject!(x1,Ph,xp,w,w_sums)
copy!(x1_mat,x1)


# ---- Same using the PatchBasedSmoother ---- #

x2_mat = PVector(0.5,Ah.cols)
r2_mat = fh-Ah*x2_mat
consistent!(r2_mat)
solve!(x2_mat,Mns,r2_mat)


# ---- Smoother inside Richardson

x3_mat = PVector(0.5,Ah.cols)
r3_mat = fh-Ah*x3_mat
consistent!(r3_mat)
solve!(x3_mat,Rns,r3_mat)
consistent!(x3_mat)

end