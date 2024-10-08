using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers

using Gridap.MultiField, Gridap.Adaptivity
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers

function get_bilinear_form(mh_lev,biform,qdegree)
  model = GridapSolvers.get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

cmodel = CartesianDiscreteModel((0,1,0,1),(10,10))
fmodel = refine(cmodel,2)
mh = ModelHierarchy([fmodel,cmodel])

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
sh = FESpace(mh,reffe;dirichlet_tags="boundary")

qdegree = 2*(order+1)
biform(u,v,dΩ) = ∫(∇(u)⋅∇(v))dΩ

biforms = map(mhl -> get_bilinear_form(mhl,biform,qdegree),mh)

smoothers = [RichardsonSmoother(JacobiLinearSolver(),10,0.2)]
prolongations = setup_prolongation_operators(
  sh,qdegree;mode=:residual
)
restrictions = setup_restriction_operators(
  sh,qdegree;mode=:residual
)

gmg = GMGLinearSolver(
  mh,sh,sh,biforms,
  prolongations,restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter=3,mode=:preconditioner,verbose=true
)

solver = CGSolver(gmg;verbose=true)

# Finest level
model = GridapSolvers.get_model(mh,1)
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)
V = get_fe_space(sh,1)
a(u,v) = biform(u,v,dΩ)
l(v) = ∫(v)dΩ

op = AffineFEOperator(a,l,V,V)
A, b = get_matrix(op), get_vector(op)

ns = numerical_setup(symbolic_setup(solver,A),A)

x = zeros(size(b))
solve!(x,ns,b)
