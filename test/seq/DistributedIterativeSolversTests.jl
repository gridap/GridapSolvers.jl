module DistributedIterativeSolversTests

using Test
using Gridap
using IterativeSolvers
using LinearAlgebra
using SparseArrays

using PartitionedArrays
using GridapSolvers
using GridapSolvers.LinearSolvers

function l2_error(uh,vh,dΩ)
  eh = uh-vh
  return sum(∫(eh⋅eh)dΩ) 
end

sol(x) = sum(x)

backend = SequentialBackend()
ranks = (1,2)
parts = get_part_ids(backend,ranks)

model = CartesianDiscreteModel(parts,(0,1,0,1),(4,8))

order  = 1
reffe  = ReferenceFE(lagrangian,Float64,order)
Vh     = TestFESpace(model,reffe;dirichlet_tags="boundary")
Uh     = TrialFESpace(Vh,sol)
Ω      = Triangulation(model)
dΩ     = Measure(Ω,2*order+1)
a(u,v) = ∫(v⋅u)*dΩ
l(v)   = ∫(1*v)*dΩ

op    = AffineFEOperator(a,l,Uh,Vh)
sol_h = solve(op)

A = get_matrix(op)
b = get_vector(op)

# CG
solver = ConjugateGradientSolver(;maxiter=100,reltol=1.e-12)
ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)

x = LinearSolvers.allocate_col_vector(A)
y = copy(b)
solve!(x,ns,y)
@test l2_error(FEFunction(Uh,x),sol_h,dΩ) < 1.e-10

# SSOR
solver = SSORSolver(2.0/3.0;maxiter=100)
ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)

x = LinearSolvers.allocate_col_vector(A)
y = copy(b)
cg!(x,A,y;verbose=true,Pl=ns)
@test l2_error(FEFunction(Uh,x),sol_h,dΩ) < 1.e-10


end