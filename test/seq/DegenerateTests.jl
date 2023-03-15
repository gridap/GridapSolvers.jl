#module DegenerateTests

using LinearAlgebra
using FillArrays
using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Adaptivity
using Test
using IterativeSolvers

include("../../src/GridapFixes.jl")

function cg_solve(op)
  A = get_matrix(op)
  b = get_vector(op)
  x = PVector(0.0,A.cols)
  IterativeSolvers.cg!(x,A,b;verbose=true,reltol=1.0e-06)
  return x
end

#function main()
  domain    = (0,1,0,1)
  partition = (0,0)
  cmodel    = CartesianDiscreteModel(domain,partition)
  model     = UnstructuredDiscreteModel(cmodel)

  order  = 1
  sol(x) = x[1] + x[2]
  reffe  = ReferenceFE(lagrangian,Float64,order)
  Vh     = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
  Uh     = TrialFESpace(sol,Vh)
  Ω      = Triangulation(model)
  dΩ     = Measure(Ω,2*order+1)

  u_sol  = interpolate(sol,Vh)
  a(u,v) = ∫(v⋅u)*dΩ
  l(v)   = ∫(v⋅u_sol)*dΩ
  op     = AffineFEOperator(a,l,Uh,Vh)

  assemble_matrix(a,Uh,Vh)

  c = a(get_trial_fe_basis(Uh),get_fe_basis(Vh))
  first(c.dict)

  dofs = get_fe_dof_basis(Uh)
  dofs(u_sol)

  x      = cg_solve(op)
  uh     = FEFunction(x,Uh)

  eh = ∫(uh-u_sol)*dΩ
  e  = sum(eh)
  println(e)
#end

main()

#end