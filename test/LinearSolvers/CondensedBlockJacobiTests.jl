module CondensedBlockJacobiTests

using Test
using LinearAlgebra

using Gridap
using Gridap.Algebra, Gridap.Geometry, Gridap.Adaptivity

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

# Groups of fine cells sharing a coarse parent, as non-overlapping patches.
function parent_patch_topology(model::AdaptedDiscreteModel)
  glue = get_adaptivity_glue(model)
  topo = get_grid_topology(model)
  return PatchTopology(topo, Gridap.Arrays.inverse_table(glue.n2o_faces_map[end]))
end

function riesz_problem(model, reffe, biform)
  V = TestFESpace(model, reffe; dirichlet_tags = "boundary")
  dΩ = Measure(Triangulation(model), 8)
  A = assemble_matrix((u,v) -> biform(u,v,dΩ), V, V)
  return V, A
end

function exactness_driver(model, reffe, biform)
  V, A = riesz_problem(model, reffe, biform)
  b = rand(size(A,1))
  ptopo = PatchTopology(ReferenceFE{0}, model)
  S = BlockJacobiSolver(V, ptopo; assembly = :star)
  x0 = solve(S, A, b)
  for block_ptopo in (PatchBasedSmoothers.cell_patch_topology(V), parent_patch_topology(model))
    for caching in (:none, :interior, :all)
      Sc = CondensedBlockJacobiSolver(V, ptopo, block_ptopo; assembly = :star, caching)
      x1 = solve(Sc, A, b)
      @test norm(x1 - x0) < 1.e-10 * norm(x0)
    end
  end
end

function gmg_driver(mh, reffe, biform, u)
  nlevs = num_levels(mh)
  tests = TestFESpace(mh, reffe; dirichlet_tags = "boundary")
  trials = TrialFESpace(tests, u)
  qdegree = 6
  f(x) = u(x) - Δ(u)(x)
  liform(v,dΩ) = ∫(v*f)dΩ
  smatrices, A, b = compute_hierarchy_matrices(trials, tests, biform, liform, qdegree)

  smoothers = map(view(tests, 1:nlevs-1)) do shl
    model = get_model(shl)
    space = get_fe_space(shl)
    ptopo = PatchTopology(ReferenceFE{0}, model)
    solver = CondensedBlockJacobiSolver(space, ptopo; assembly = :star)
    return RichardsonSmoother(solver, 5, 0.2)
  end
  restrictions, prolongations = setup_transfer_operators(tests, qdegree; mode = :residual)

  gmg = GMGLinearSolver(
    smatrices, prolongations, restrictions;
    pre_smoothers = smoothers, post_smoothers = smoothers,
    coarsest_solver = LUSolver(), maxiter = 1, mode = :preconditioner
  )
  solver = CGSolver(gmg; maxiter = 20, atol = 1e-14, rtol = 1.e-8, verbose = true)
  ns = numerical_setup(symbolic_setup(solver, A), A)
  x = zeros(size(A,2))
  solve!(x, ns, b)

  model = get_model(mh, 1)
  dΩ = Measure(Triangulation(model), qdegree)
  eh = u - FEFunction(get_fe_space(trials, 1), x)
  @test sqrt(sum(∫(eh*eh)dΩ)) < 1.e-8
  @test solver.log.num_iters < 20
end

function main(D)
  biform(u,v,dΩ) = ∫(u*v + ∇(u)⋅∇(v))dΩ
  hdiv_biform(u,v,dΩ) = ∫(u⋅v + divergence(u)*divergence(v))dΩ

  cmodel = CartesianDiscreteModel(Tuple(repeat([0,1], D)), Tuple(fill(2, D)))
  model1 = refine(cmodel)
  model2 = refine(model1)
  exactness_driver(model1, ReferenceFE(lagrangian, Float64, (D == 2) ? 3 : 2), biform)
  exactness_driver(model1, ReferenceFE(raviart_thomas, Float64, 1), hdiv_biform)

  mh = ModelHierarchy([model2, model1, cmodel])
  u(x) = (D == 2) ? x[1]^2 + x[2]^2 : x[1]^2 + x[2]^2 + x[3]^2
  gmg_driver(mh, ReferenceFE(lagrangian, Float64, 2), biform, u)
end

end # module CondensedBlockJacobiTests
