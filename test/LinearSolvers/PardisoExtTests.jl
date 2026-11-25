module PardisoExtTests

using Test
using Pardiso

using Gridap, Gridap.Algebra
using LinearAlgebra
using PartitionedArrays
using GridapDistributed
using GridapSolvers

function main()
  model = CartesianDiscreteModel((0,1,0,1),(10,10))

  reffe = ReferenceFE(lagrangian,Float64,1)
  V = FESpace(model,reffe;dirichlet_tags=["boundary"])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)

  f(x) = sin(2π*x[1])*sin(2π*x[2])
  a(u,v) = ∫(∇(u)⋅∇(v))dΩ
  l(v) = ∫(f*v)dΩ

  op = AffineFEOperator(a,l,V,V)
  A, b = Algebra.get_matrix(op), Algebra.get_vector(op)
  x = A \ b

  ps = PardisoLinearSolver(;nthreads=Threads.nthreads(),verbose=true)
  ns = numerical_setup(symbolic_setup(ps,A),A)
  y = allocate_in_domain(A)
  Algebra.solve!(y,ns,b)
  
  @test norm(x - y) < 1e-10

  ns = numerical_setup(symbolic_setup(PardisoLinearSolver(;nthreads=Threads.nthreads()),A),A)
  ps, ps_mat = ns.solver.ps, ns.ps_mat
  ns = nothing
  GC.gc(); GC.gc()
  z = allocate_in_domain(A)
  Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
  Pardiso.pardiso(ps, z, ps_mat, b)
  @test norm(x - z) < 1e-10
end

end
