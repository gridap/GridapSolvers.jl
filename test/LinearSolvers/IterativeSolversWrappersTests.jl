module IterativeSolversWrappersTests

using Test
using Gridap, Gridap.Algebra
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using PartitionedArrays

using GridapDistributed
using GridapSolvers
using GridapSolvers.LinearSolvers

sol(x) = x[1] + x[2]
f(x)   = -Δ(sol)(x)

function test_solver(solver,op,Uh,dΩ)
  A, b = get_matrix(op), get_vector(op);
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = allocate_in_domain(A); fill!(x,zero(eltype(x)))
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

  verbose = i_am_main(parts)
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

  verbose && println("> Testing CG")
  cg_solver = IS_ConjugateGradientSolver(;maxiter=100,reltol=1.e-12,verbose=verbose)
  test_solver(cg_solver,op,Uh,dΩ)

  if prod(np) == 1
    verbose && println("> Testing SSOR")
    ssor_solver = IS_SSORSolver(2.0/3.0;maxiter=1000)
    test_solver(ssor_solver,op,Uh,dΩ)

    verbose && println("> Testing GMRES")
    gmres_solver = IS_GMRESSolver(;maxiter=100,reltol=1.e-12,verbose=verbose)
    test_solver(gmres_solver,op,Uh,dΩ)

    verbose && println("> Testing MINRES")
    minres_solver = IS_MINRESSolver(;maxiter=100,reltol=1.e-12,verbose=verbose)
    test_solver(minres_solver,op,Uh,dΩ)
  end
end

end