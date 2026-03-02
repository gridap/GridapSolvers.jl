module KrylovTests

using Test
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

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

  # GMRES with left and right preconditioner
  gmres = LinearSolvers.GMRESSolver(40;Pr=P,Pl=P,rtol=1.e-8,verbose=verbose)
  test_solver(gmres,op,Uh,dΩ)

  # GMRES without preconditioner
  gmres = LinearSolvers.GMRESSolver(10;rtol=1.e-8,verbose=verbose)
  test_solver(gmres,op,Uh,dΩ)

  gmres = LinearSolvers.GMRESSolver(10;restart=true,rtol=1.e-8,verbose=verbose)
  test_solver(gmres,op,Uh,dΩ)

  fgmres = LinearSolvers.FGMRESSolver(10,P;rtol=1.e-8,verbose=verbose)
  test_solver(fgmres,op,Uh,dΩ)

  fgmres = LinearSolvers.FGMRESSolver(10,P;restart=true,rtol=1.e-8,verbose=verbose)
  test_solver(fgmres,op,Uh,dΩ)

  cg = LinearSolvers.CGSolver(;rtol=1.e-8,verbose=verbose)
  test_solver(cg,op,Uh,dΩ)

  pcg = LinearSolvers.CGSolver(P;rtol=1.e-8,verbose=verbose)
  test_solver(pcg,op,Uh,dΩ)

  fpcg = LinearSolvers.CGSolver(P;flexible=true,rtol=1.e-8,verbose=verbose)
  test_solver(fpcg,op,Uh,dΩ)

  minres = LinearSolvers.MINRESSolver(;Pl=P,rtol=1.e-8,verbose=verbose)
  test_solver(minres,op,Uh,dΩ)
end

nvec = [4,10,20,100,200]
ρvec = Float64[]
for n in nvec
  model = CartesianDiscreteModel((0,1,0,1),(n,n))

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
  verbose = true

  maxiter = 1000
  cg = LinearSolvers.CGSolver(P;rtol=1.e-12,maxiter=maxiter,verbose=verbose,diagnostic=LinearSolvers.LanczosDiagnostic(maxiter+1))
  
  A = get_matrix(op)
  b = randn(size(A,1))
  ns = numerical_setup(symbolic_setup(cg,A),A)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)

  # using LinearAlgebra
  # diag = cg.diag
  # k = diag.k[]
  # M = SymTridiagonal(view(diag.delta,1:k), view(diag.gamma,2:k))
  # λ = eigvals(M)

  ρ = LinearSolvers.estimate!(cg.diag)
  push!(ρvec,ρ)
end

hvec = 1 ./ nvec
ρvec


end