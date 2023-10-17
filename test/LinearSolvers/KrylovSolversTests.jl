module KrylovSolversTests

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

  x = allocate_col_vector(A)
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

  gmres = LinearSolvers.GMRESSolver(40;Pr=P,Pl=P,rtol=1.e-8,verbose=verbose)
  test_solver(gmres,op,Uh,dΩ)

  fgmres = LinearSolvers.FGMRESSolver(40,P;rtol=1.e-8,verbose=verbose)
  test_solver(fgmres,op,Uh,dΩ)

  pcg = LinearSolvers.CGSolver(P;rtol=1.e-8,verbose=verbose)
  test_solver(pcg,op,Uh,dΩ)

  fpcg = LinearSolvers.CGSolver(P;flexible=true,rtol=1.e-8,verbose=verbose)
  test_solver(fpcg,op,Uh,dΩ)

  minres = LinearSolvers.MINRESSolver(;Pl=P,Pr=P,rtol=1.e-8,verbose=verbose)
  test_solver(minres,op,Uh,dΩ)
end

end