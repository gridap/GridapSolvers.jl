module IterativeSolversTests

using Test
using Gridap
using IterativeSolvers
using LinearAlgebra
using SparseArrays
using PartitionedArrays

using GridapSolvers
using GridapSolvers.LinearSolvers

sol(x) = x[1] + x[2]
f(x)   = -Δ(sol)(x)

function l2_error(x,Uh,dΩ)
  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  return sum(∫(eh*eh)*dΩ)
end

function main(model,is_distributed)
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

  # CG
  solver = IS_ConjugateGradientSolver(;maxiter=100,reltol=1.e-12)
  ss = symbolic_setup(solver,A)
  ns = numerical_setup(ss,A)

  x = LinearSolvers.allocate_col_vector(A)
  y = copy(b)
  solve!(x,ns,y)
  @test l2_error(x,Uh,dΩ) < 1.e-8

  # SSOR
  solver = IS_SSORSolver(2.0/3.0;maxiter=100)
  ss = symbolic_setup(solver,A)
  ns = numerical_setup(ss,A)

  x = LinearSolvers.allocate_row_vector(A)
  y = copy(b)
  solve!(x,ns,y)
  !is_distributed && (@test l2_error(x,Uh,dΩ) < 1.e-8)

    if !is_distributed
    # GMRES
    solver = IS_GMRESSolver(;maxiter=100,reltol=1.e-12)
    ss = symbolic_setup(solver,A)
    ns = numerical_setup(ss,A)

    x = LinearSolvers.allocate_row_vector(A)
    y = copy(b)
    solve!(x,ns,y)
    @test l2_error(x,Uh,dΩ) < 1.e-8

    # MINRES
    solver = IS_MINRESSolver(;maxiter=100,reltol=1.e-12)
    ss = symbolic_setup(solver,A)
    ns = numerical_setup(ss,A)

    x = LinearSolvers.allocate_row_vector(A)
    y = copy(b)
    solve!(x,ns,y)
    @test l2_error(x,Uh,dΩ) < 1.e-8
  end
end

# Completely serial
partition = (8,8)
domain = (0,1,0,1)
model  = CartesianDiscreteModel(domain,partition)
main(model,false)

# Sequential
backend = SequentialBackend()
ranks = (1,2)
parts = get_part_ids(backend,ranks)

model  = CartesianDiscreteModel(parts,domain,partition)
main(model,true)

end