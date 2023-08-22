module SymGaussSeidelSmoothersTests

using Test
using MPI
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

  P  = SymGaussSeidelSmoother(10)
  ss = symbolic_setup(P,A)
  ns = numerical_setup(ss,A)

  x = LinearSolvers.allocate_col_vector(A)
  x, history = IterativeSolvers.cg!(x,A,b;
                                    verbose=true,
                                    reltol=1.0e-8,
                                    Pl=ns,
                                    log=true);

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  return E < 1.e-8
end

# Completely serial
mesh_partition = (8,8)
domain = (0,1,0,1)
model  = CartesianDiscreteModel(domain,mesh_partition)
@test main(model)

# Sequential
num_ranks = (1,2)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end

model  = CartesianDiscreteModel(parts,num_ranks,domain,mesh_partition)
@test main(model)

end