
using Gridap
using GridapSolvers

using Gridap.Adaptivity
using GridapSolvers.MultilevelTools
using GridapSolvers.LinearSolvers

u(x) = x[1] + x[2]

nlevs = 3
models = Vector{DiscreteModel}(undef,nlevs)
models[end] = CartesianDiscreteModel((0,1,0,1),(4,4))
for lev in 2:-1:1
  models[lev] = refine(models[lev+1])
end
mh = ModelHierarchy(models)

order = 1
reffe = ReferenceFE(lagrangian,Float64,1)
tests = FESpace(mh, reffe; dirichlet_tags="boundary")
trials = TrialFESpace(tests, u)

qdegree   = 2*order+1

f(x) = -Δ(u)(x)
biform(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ
liform(v,dΩ)   = ∫(v*f)dΩ

mats, A, b = compute_hierarchy_matrices(trials,tests,biform,liform,qdegree)
restrictions, prolongations = setup_transfer_operators(tests, qdegree; mode=:residual)

n, ω = 1, 1.
sym_smoothers = fill(GaussSeidelSmoother(n,ω,:symmetric),nlevs-1)
fw_smoothers  = fill(GaussSeidelSmoother(n,ω,:forward),nlevs-1)
bw_smoothers  = fill(GaussSeidelSmoother(n,ω,:backward),nlevs-1)

gmg = GMGLinearSolver(
  mats, prolongations,restrictions,
  pre_smoothers = fw_smoothers,
  post_smoothers = bw_smoothers,
  coarsest_solver = LUSolver(),
  maxiter=1, mode=:preconditioner,
  verbose = true
)
gmg.log.depth = 2

solver = CGSolver(gmg;maxiter=20,atol=1e-14,rtol=1.e-6,verbose=true)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = fill(0.0,axes(A,2))
solve!(x,ns,b)
