using Test
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

using GridapSolvers
using GridapSolvers.LinearSolvers

sol(x) = x[1] + x[2]
f(x)   = -Δ(sol)(x)

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
