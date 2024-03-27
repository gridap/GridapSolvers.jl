module DistributedPatchFESpacesTests

using LinearAlgebra
using Test
using PartitionedArrays
using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.ReferenceFEs
using GridapDistributed
using FillArrays

using GridapSolvers
import GridapSolvers.PatchBasedSmoothers as PBS

np = (1,1)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

domain = (0,1,0,1)
domain_partition = (2,2)#(2,4)

if prod(np) == 1
  model = CartesianDiscreteModel(domain,domain_partition)
else
  model = CartesianDiscreteModel(parts,ranks,domain,domain_partition)
end
pbs = PBS.PatchBoundaryExclude()
PD  = PBS.PatchDecomposition(model;patch_boundary_style=pbs)

conf = :hdiv
order = 1
if conf === :h1
  _reffe = ReferenceFE(lagrangian,Float64,order)
  reffe = LagrangeRefFE(Float64,QUAD,order)
  conformity = H1Conformity()
  f = 1.0
  u_bc = 0.0
  biform(u,v,dΩ) = ∫(v⋅u)*dΩ
  liform(v,dΩ)   = ∫(f⋅v)*dΩ
else
  _reffe = ReferenceFE(raviart_thomas,Float64,order)
  reffe = RaviartThomasRefFE(Float64,QUAD,order)
  conformity = DivConformity()
  f = VectorValue(1.0,1.0)
  u_bc = VectorValue(1.0,0.0)
  biform(u,v,dΩ) = ∫(u⋅v + divergence(v)⋅divergence(u))*dΩ
  liform(v,dΩ)   = ∫(f⋅v)*dΩ
end
Vh = TestFESpace(model,_reffe,dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u_bc)
Ph = PBS.PatchFESpace(model,_reffe,conformity,PD,Vh)

Ωₚ  = Triangulation(PD)
dΩₚ = Measure(Ωₚ,2*(order+2))
ap(u,v) = biform(u,v,dΩₚ)
Ahp = assemble_matrix(ap,Ph,Ph)
det(Ahp)

n1 = 4 # 8
A_corner = Matrix(Ahp)[1:n1,1:n1]
det(A_corner)

# ---- Testing Prolongation and Injection ---- #

w, w_sums = PBS.compute_weight_operators(Ph,Vh);

xP = zero_free_values(Ph) .+ 1.0
yP = zero_free_values(Ph)
x  = zero_free_values(Vh) .+ 1.0
y  = zero_free_values(Vh)

PBS.prolongate!(yP,Ph,x)
PBS.inject!(y,Ph,yP,w,w_sums)
@test x ≈ y

PBS.inject!(x,Ph,xP,w,w_sums)
PBS.prolongate!(yP,Ph,x)
@test xP ≈ yP

@benchmark PBS.inject!($y,$Ph,$yP,$w,$w_sums)

@benchmark PBS.inject!($y,$Ph,$yP)

# ---- Assemble systems ---- #

Ω  = Triangulation(model)
dΩ = Measure(Ω,2*order+1)
a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)

assembler = SparseMatrixAssembler(Vh,Vh)
Ah = assemble_matrix(a,assembler,Vh,Vh)
fh = assemble_vector(l,assembler,Vh)

sol_h = solve(LUSolver(),Ah,fh)

Ωₚ  = Triangulation(PD)
dΩₚ = Measure(Ωₚ,2*order+1)
ap(u,v) = biform(u,v,dΩₚ)
lp(v) = liform(v,dΩₚ)

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

x1_mat = pfill(0.5,partition(axes(Ah,2)))
r1_mat = fh-Ah*x1_mat
consistent!(r1_mat) |> fetch

r1 = pfill(0.0,partition(Vh.gids))
x1 = pfill(0.0,partition(Vh.gids))
rp = pfill(0.0,partition(Ph.gids))
xp = pfill(0.0,partition(Ph.gids))
rp_mat = pfill(0.0,partition(axes(Ahp,2)))
xp_mat = pfill(0.0,partition(axes(Ahp,2)))

copy!(r1,r1_mat)
consistent!(r1) |> fetch
PBS.prolongate!(rp,Ph,r1)

copy!(rp_mat,rp)
solve!(xp_mat,LUns,rp_mat)
copy!(xp,xp_mat)

w, w_sums = PBS.compute_weight_operators(Ph,Vh);
PBS.inject!(x1,Ph,xp,w,w_sums)
copy!(x1_mat,x1)


# ---- Same using the PatchBasedSmoother ---- #

x2_mat = pfill(0.5,partition(axes(Ah,2)))
r2_mat = fh-Ah*x2_mat
consistent!(r2_mat) |> fetch
solve!(x2_mat,Mns,r2_mat)


# ---- Smoother inside Richardson

x3_mat = pfill(0.5,partition(axes(Ah,2)))
r3_mat = fh-Ah*x3_mat
consistent!(r3_mat) |> fetch
solve!(x3_mat,Rns,r3_mat)
consistent!(x3_mat) |> fetch

end