module GMRESSolversTests

using Test
using Gridap
using GridapDistributed
using PartitionedArrays
using IterativeSolvers

using GridapSolvers
using GridapSolvers.LinearSolvers

sol(x) = x[1] + x[2]
f(x)   = -Δ(sol)(x)

function test_solver(solver,op,Uh,dΩ)
  A, b = get_matrix(op), get_vector(op);
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = LinearSolvers.allocate_col_vector(A)
  solve!(x,ns,b)

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  @test E < 1.e-8
end

function main(model)
  order  = 1
  qorder = order*2 + 1
  reffe  = ReferenceFE(lagrangian,Float64,order)
  Vh     = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  Uh     = TrialFESpace(Vh,sol)
  u      = interpolate(sol,Uh)

  Ω      = Triangulation(model)
  dΩ     = Measure(Ω,qorder)
  a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
  l(v)   = ∫(v⋅f)*dΩ
  op = AffineFEOperator(a,l,Uh,Vh)
  
  P = JacobiLinearSolver()

  gmres = LinearSolvers.GMRESSolver(40;Pr=P,Pl=P,rtol=1.e-8,verbose=true)
  test_solver(gmres,op,Uh,dΩ)

  fgmres = LinearSolvers.FGMRESSolver(40,P;rtol=1.e-8,verbose=true)
  test_solver(fgmres,op,Uh,dΩ)

  pcg = LinearSolvers.CGSolver(P;rtol=1.e-8,verbose=true)
  test_solver(pcg,op,Uh,dΩ)

  fpcg = LinearSolvers.CGSolver(P;flexible=true,rtol=1.e-8,verbose=true)
  test_solver(fpcg,op,Uh,dΩ)

  minres = LinearSolvers.MINRESSolver(;rtol=1.e-8,verbose=true)
  test_solver(minres,op,Uh,dΩ)
end

# Completely serial
mesh_partition = (10,10)
domain = (0,1,0,1)
model  = CartesianDiscreteModel(domain,mesh_partition)
main(model)

# Sequential
num_ranks = (1,2)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end

model  = CartesianDiscreteModel(parts,num_ranks,domain,mesh_partition)
main(model)

end