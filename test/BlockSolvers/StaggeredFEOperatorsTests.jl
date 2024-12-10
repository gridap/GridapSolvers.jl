module StaggeredFEOperatorsTests

using Test
using Gridap, GridapDistributed, PartitionedArrays
using Gridap.MultiField
using BlockArrays
using GridapSolvers
using GridapSolvers.BlockSolvers

model = CartesianDiscreteModel((0,1,0,1),(4,4))

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe;dirichlet_tags="boundary")

sol = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> x[1] - x[2]]
U1 = TrialFESpace(V,sol[1])
U2 = TrialFESpace(V,sol[2])
U3 = TrialFESpace(V,sol[3])
U4 = TrialFESpace(V,sol[4])

mfs = BlockMultiFieldStyle(3,(1,2,1))
X = MultiFieldFESpace([U1,U2,U3,U4];style=mfs)
Y = MultiFieldFESpace([V,V,V,V];style=mfs)

Ω = Triangulation(model)
dΩ = Measure(Ω,3*order)

a1((),u1,v1) = ∫(u1 * v1)dΩ
l1((),v1) = ∫(sol[1] * v1)dΩ

a2((u1,),(u2,u3),(v2,v3)) = ∫(u1 * u2 * v2)dΩ + ∫(u3 * v3)dΩ
l2((u1,),(v2,v3)) = ∫(sol[2] * u1 * v2)dΩ + ∫(sol[3] * v3)dΩ

a3((u1,(u2,u3)),u4,v4) = ∫((u1 + u2) * u4 * v4)dΩ
l3((u1,(u2,u3)),v4) = ∫(sol[4] * (u1 + u2) * v4)dΩ

UB1, VB1 = U1, V
UB2, VB2 = MultiFieldFESpace([U2,U3]), MultiFieldFESpace([V,V])
UB3, VB3 = U4, V
op = StaggeredAffineFEOperator([a1,a2,a3],[l1,l2,l3],[UB1,UB2,UB3],[VB1,VB2,VB3])

solver = StaggeredFESolver([LUSolver(),LUSolver(),LUSolver()])
xh = solve(solver,op)

xh_exact = interpolate(sol,X)

for k in 1:4
  eh_k = xh[k] - xh_exact[k]
  e_k = sum(∫(eh_k * eh_k)dΩ)
  println("Error in field $k: $e_k")
  @test e_k < 1e-10
end

end # module