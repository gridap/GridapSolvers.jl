module StaggeredFEOperatorsTests

using Test
using Gridap, GridapDistributed, PartitionedArrays
using Gridap.MultiField
using BlockArrays
using GridapSolvers
using GridapSolvers.BlockSolvers, GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers

function test_solution(xh,sol,X,dΩ)
  N = length(sol)
  xh_exact = interpolate(sol,X)
  for k in 1:N
    eh_k = xh[k] - xh_exact[k]
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
    println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end
end

function driver_affine(model,verbose)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model,reffe;dirichlet_tags="boundary")

  sol = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> x[1] - x[2]]
  U1 = TrialFESpace(V,sol[1])
  U2 = TrialFESpace(V,sol[2])
  U3 = TrialFESpace(V,sol[3])
  U4 = TrialFESpace(V,sol[4])

  # Define weakforms
  Ω = Triangulation(model)
  dΩ = Measure(Ω,3*order)

  a1((),u1,v1) = ∫(u1 * v1)dΩ
  l1((),v1) = ∫(sol[1] * v1)dΩ

  a2((u1,),(u2,u3),(v2,v3)) = ∫(u1 * u2 * v2)dΩ + ∫(u3 * v3)dΩ
  l2((u1,),(v2,v3)) = ∫(sol[2] * u1 * v2)dΩ + ∫(sol[3] * v3)dΩ

  a3((u1,(u2,u3)),u4,v4) = ∫((u1 + u2) * u4 * v4)dΩ
  l3((u1,(u2,u3)),v4) = ∫(sol[4] * (u1 + u2) * v4)dΩ

  # Define solver: Each block will be solved with a CG solver
  lsolver = CGSolver(JacobiLinearSolver();rtol=1.e-10,verbose=verbose)
  solver = StaggeredFESolver(fill(lsolver,3))

  # Create operator from full spaces
  mfs = BlockMultiFieldStyle(3,(1,2,1))
  X = MultiFieldFESpace([U1,U2,U3,U4];style=mfs)
  Y = MultiFieldFESpace([V,V,V,V];style=mfs)
  op = StaggeredAffineFEOperator([a1,a2,a3],[l1,l2,l3],X,Y)
  xh = solve(solver,op)
  test_solution(xh,sol,X,dΩ)

  # Create operator from components
  UB1, VB1 = U1, V
  UB2, VB2 = MultiFieldFESpace([U2,U3]), MultiFieldFESpace([V,V])
  UB3, VB3 = U4, V
  op = StaggeredAffineFEOperator([a1,a2,a3],[l1,l2,l3],[UB1,UB2,UB3],[VB1,VB2,VB3])

  # Solve and keep caches for reuse
  xh = zero(X)
  xh, cache = solve!(xh,solver,op);
  xh, cache = solve!(xh,solver,op,cache);
  test_solution(xh,sol,X,dΩ)

  return true
end



############################################################################################

np = (2,2)
ranks = DebugArray(LinearIndices((prod(np),)))
verbose = i_am_main(ranks)
model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4))

@testset "StaggeredAffineFEOperators" driver_affine(model,verbose)

model = CartesianDiscreteModel((0,1,0,1),(4,4))

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe;dirichlet_tags="boundary")

sol = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> x[1] - x[2]]
U1 = TrialFESpace(V,sol[1])
U2 = TrialFESpace(V,sol[2])
U3 = TrialFESpace(V,sol[3])
U4 = TrialFESpace(V,sol[4])

# Define weakforms
Ω = Triangulation(model)
dΩ = Measure(Ω,4*order)

F(u::Function) = x -> (u(x) + 1) * u(x)
F(u) = (u + 1) * u
dF(u,du) = 2.0 * u * du + du

j1((),u1,du1,dv1) = ∫(dF(u1,du1) * dv1)dΩ
r1((),u1,v1) = ∫((F(u1) - F(sol[1])) * v1)dΩ

j2((u1,),(u2,u3),(du2,du3),(dv2,dv3)) = ∫(u1 * dF(u2,du2) * dv2)dΩ + ∫(dF(u3,du3) * dv3)dΩ
r2((u1,),(u2,u3),(v2,v3)) = ∫(u1 * (F(u2) - F(sol[2])) * v2)dΩ + ∫((F(u3) - F(sol[3])) * v3)dΩ

j3((u1,(u2,u3)),u4,du4,dv4) = ∫(u3 * dF(u4,du4) * dv4)dΩ
r3((u1,(u2,u3)),u4,v4) = ∫(u3 * (F(u4) - F(sol[4])) * v4)dΩ

# Define solver: Each block will be solved with a LU solver
lsolver = LUSolver()
nlsolver = NewtonSolver(lsolver;rtol=1.e-10,verbose=verbose)
solver = StaggeredFESolver(fill(nlsolver,3))

# Create operator from full spaces
mfs = BlockMultiFieldStyle(3,(1,2,1))
X = MultiFieldFESpace([U1,U2,U3,U4];style=mfs)
Y = MultiFieldFESpace([V,V,V,V];style=mfs)
op = StaggeredNonlinearFEOperator([r1,r2,r3],[j1,j2,j3],X,Y)
xh = solve(solver,op)
test_solution(xh,sol,X,dΩ)

# Create operator from components
UB1, VB1 = U1, V
UB2, VB2 = MultiFieldFESpace([U2,U3]), MultiFieldFESpace([V,V])
UB3, VB3 = U4, V
op = StaggeredNonlinearFEOperator([r1,r2,r3],[j1,j2,j3],[UB1,UB2,UB3],[VB1,VB2,VB3])

xh = zero(X)
xh, cache = solve!(xh,solver,op);
test_solution(xh,sol,X,dΩ)

end # module