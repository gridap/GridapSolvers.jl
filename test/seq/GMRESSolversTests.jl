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
  A, b = get_matrix(op), get_vector(op);

  Pl = JacobiLinearSolver()
  solver = LinearSolvers.GMRESSolver(40,Pl,1.e-8)
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = LinearSolvers.allocate_col_vector(A)
  solve!(x,ns,b)

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  return E < 1.e-8
end

# Completely serial
mesh_partition = (10,10)
domain = (0,1,0,1)
model  = CartesianDiscreteModel(domain,mesh_partition)
@test main(model)

# Sequential
num_ranks = (1,2)
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end

model  = CartesianDiscreteModel(parts,num_ranks,domain,mesh_partition)
@test main(model)

end