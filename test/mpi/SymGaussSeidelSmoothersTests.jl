module RichardsonSmoothersTests

using Test
using MPI
using Gridap
using GridapDistributed
using PartitionedArrays
using IterativeSolvers

using GridapSolvers
using GridapSolvers.LinearSolvers

function main(parts,partition)
  domain = (0,1,0,1)
  model  = CartesianDiscreteModel(parts,domain,partition)

  sol(x) = x[1] + x[2]
  f(x)   = -Δ(sol)(x)

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
  A, b = get_matrix(op), get_vector(op)

  P  = SymGaussSeidelSmoother(10)
  ss = symbolic_setup(P,A)
  ns = numerical_setup(ss,A)

  x = PVector(1.0,A.cols)
  x, history = IterativeSolvers.cg!(x,A,b;
                                    verbose=i_am_main(parts),
                                    reltol=1.0e-8,
                                    Pl=ns,
                                    log=true)

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  if i_am_main(parts)
    println("L2 Error: ", E)
  end
  
  @test E < 1.e-8
end

partition = (32,32)
ranks = (2,2)

with_backend(main,MPIBackend(),ranks,partition)
MPI.Finalize()

end