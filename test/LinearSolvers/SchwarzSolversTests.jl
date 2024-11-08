module SchwarzSolversTests

using Test
using MPI
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

using GridapSolvers
using GridapSolvers.LinearSolvers

function main(distribute,np)
  sol(x) = x[1] + x[2]
  f(x)   = -Δ(sol)(x)

  parts = distribute(LinearIndices((prod(np),)))
  model = CartesianDiscreteModel(parts,np,(0,1,0,1),(8,8))

  order  = 1
  qorder = order*2 + 1
  reffe  = ReferenceFE(lagrangian,Float64,order)
  Vh     = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  Uh     = TrialFESpace(Vh,sol)
  u      = interpolate(sol,Uh)

  Ω      = Triangulation(model)
  dΩ     = Measure(Ω,qorder)
  a(u,v) = ∫(v⋅u)*dΩ
  l(v)   = ∫(v⋅sol)*dΩ

  op = AffineFEOperator(a,l,Uh,Vh)
  A, b = get_matrix(op), get_vector(op)

  P = SchwarzLinearSolver(LUSolver())
  solver = CGSolver(P;rtol=1.0e-8,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)
  x = allocate_in_domain(A); fill!(x,zero(eltype(x)))
  solve!(x,ns,b)

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  if i_am_main(parts)
    println("L2 Error: ", E)
  end
  
  @test E < 1.e-8
end

end # module