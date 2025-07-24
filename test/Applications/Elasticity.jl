module PETScElasticitySolverTests

using Gridap
using Gridap.Geometry, Gridap.Algebra

using PartitionedArrays
using GridapDistributed
using GridapPETSc

using GridapSolvers

function main(distribute,np)
  ranks = distribute(LinearIndices((prod(np),)))
  model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(20,20))

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"diri_0",[1,3,7])
  add_tag_from_tags!(labels,"diri_1",[2,4,8])

  order = 1
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  V = TestFESpace(model,reffe,dirichlet_tags=["diri_0", "diri_1"])

  disp_x = 0.5
  g0 = VectorValue(0.0,0.0)
  g1 = VectorValue(disp_x,0.0)
  U = TrialFESpace(V,[g0,g1])

  λ = 100.0
  μ = 1.0
  σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(u,v) = ∫(ε(v) ⊙ (σ∘ε(u)))*dΩ
  l(v) = 0

  op = AffineFEOperator(a,l,U,V)

  options = "-ksp_error_if_not_converged true -ksp_converged_reason"
  x = GridapPETSc.with(args=split(options)) do
    solver = PETScElasticitySolver(U)
    A, b = get_matrix(op), get_vector(op)
    ns = numerical_setup(symbolic_setup(solver,A),A)
    x = allocate_in_domain(A)
    fill!(x,0.0)
    solve!(x,ns,b)
  end

  uh = FEFunction(U,x)
end

end # module