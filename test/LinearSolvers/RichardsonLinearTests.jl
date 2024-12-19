module RichardsonLinearTests

using Test
using Gridap, Gridap.Algebra
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

  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  @test E < 1.e-6
end

function get_mesh(parts,np)
  Dc = length(np)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (8,8)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (8,8,8)
  end
  if prod(np) == 1
    model = CartesianDiscreteModel(domain,nc)
  else
    model = CartesianDiscreteModel(parts,np,domain,nc)
  end
  return model
end

function main(distribute,np)
  parts = distribute(LinearIndices((prod(np),)))
  model = get_mesh(parts,np)
  
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
  verbose = i_am_main(parts)

  # RichardsonLinearSolver with a left preconditioner
  solver = RichardsonLinearSolver(0.5,1000; Pl = P, rtol = 1e-8, verbose = verbose)
  test_solver(solver,op,Uh,dΩ)

  # RichardsonLinearSolver without a preconditioner
  solver = RichardsonLinearSolver(0.5,1000; rtol = 1e-8, verbose = verbose)
  test_solver(solver,op,Uh,dΩ)
end

end