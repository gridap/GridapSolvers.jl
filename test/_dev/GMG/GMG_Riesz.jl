
using LinearAlgebra
using FillArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra, Gridap.Adaptivity
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers

function get_patch_smoothers(mh,tests,biform,qdegree;w=0.2)
  nlevs = num_levels(mh)
  smoothers = map(view(mh,1:nlevs-1),view(tests,1:nlevs-1)) do mhl,tests
    model = get_model(mhl)
    Vh = get_fe_space(tests)
    ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
    Ω  = Geometry.PatchTriangulation(model,ptopo)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    solver = PatchBasedSmoothers.PatchSolver(ptopo,Vh,Vh,ap;assembly=:star,collect_factorizations=false)
    if w > 0.0
      return RichardsonSmoother(solver,10,w)
    else
      return FGMRESSolver(10,solver;maxiter=10)
    end
  end
  return smoothers
end

function get_patch_smoothers_old(mh,tests,biform,qdegree;w=0.2)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),view(mh,1:nlevs-1)) do tests, mhl
    model = get_model(mhl)
    Vh = get_fe_space(tests)
    PD = PatchDecomposition(model)
    Ph = PatchFESpace(Vh,PD)
    Ω  = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    solver = PatchBasedLinearSolver(ap,Ph,Vh)
    if w > 0.0
      return RichardsonSmoother(solver,10,w)
    else
      return FGMRESSolver(10,solver;maxiter=10)
    end
  end
  return smoothers
end

function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

np = (2,2)
np_per_level = [np,np]
parts = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

nc = (2,2)
Dc = length(nc)
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
mh = CartesianModelHierarchy(parts,np_per_level,domain,nc)

cmodel = CartesianDiscreteModel((0,1,0,1),(4,4))
fmodel = refine(cmodel)
mh = ModelHierarchy([fmodel,cmodel])


model = get_model(mh,1)

order = 2
qdegree = 2*order

reffe_h1 = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_hdiv = ReferenceFE(raviart_thomas,Float64,order-1)
reffe_hcurl = ReferenceFE(nedelec,Float64,order-1)

reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Π_Qh = LocalProjectionMap(divergence,reffe_p,qdegree)

α = 1.e2
biform_h1(u,v,dΩ) = ∫(∇(u)⊙∇(v) + α * Π_Qh(u)⋅divergence(v))dΩ
biform_hdiv(u,v,dΩ) = ∫(u⋅v + α * divergence(u)⋅divergence(v))dΩ
biform_hcurl(u,v,dΩ) = ∫(u⋅v + α * curl(u)⋅curl(v))dΩ
graddiv(u,v,dΩ)  = ∫(α*(∇⋅v)⋅Π_Qh(u))dΩ

u_h1(x) = ifelse(Dc==2, VectorValue(x[1]+x[2],-x[2]) , VectorValue(x[1]+x[2],-x[2],0.0))
u_hdiv(x) = ifelse(Dc==2, VectorValue(x[1]+x[2],-x[2]) , VectorValue(x[1]+x[2],-x[2],0.0))
u_hcurl(x) = ifelse(Dc==2, VectorValue(x[1]+x[2],-x[2]) , VectorValue(x[1]+x[2],-x[2],0.0))

conformity = :H1 # :Hdiv, :Hcurl
if conformity == :H1
  biform = biform_h1
  reffe = reffe_h1
  u_exact = u_h1
elseif conformity == :Hdiv
  biform = biform_hdiv
  reffe = reffe_hdiv
  u_exact = u_hdiv
elseif conformity == :Hcurl
  biform = biform_hcurl
  reffe = reffe_hcurl
  u_exact = u_hcurl
end

#f = ifelse(Dc==2,VectorValue(1.0,1.0),VectorValue(1.0,1.0,1.0))
f = u_exact
liform(v,dΩ) = ∫(v⋅f)dΩ

tests  = TestFESpace(mh,reffe,dirichlet_tags=["tag_5"]);
trials = TrialFESpace(tests,[u_exact]);
U, V = get_fe_space(trials,1), get_fe_space(tests,1)

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)
a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,U,V)
A, b = get_matrix(op), get_vector(op);

biforms = map(mhl -> get_bilinear_form(mhl,biform,qdegree), mh)

w = 0.2
smoothers = get_patch_smoothers(
  mh,tests,biform,qdegree;w
)
smoothers_old = get_patch_smoothers_old(
  mh,tests,biform,qdegree;w
)
prolongations = setup_prolongation_operators(
  tests,qdegree;mode=:residual
)
restrictions = setup_restriction_operators(
  tests,qdegree;mode=:residual
)

patch_prolongations = setup_patch_prolongation_operators(
  tests,biform,biform,qdegree;is_nonlinear=false,collect_factorizations=false
)
patch_prolongations_old = setup_patch_prolongation_operators_old(
  tests,biform,biform,qdegree
)
patch_restrictions_old = setup_patch_restriction_operators_old(
  tests,patch_prolongations_old,biform,qdegree
)

gmg = GMGLinearSolver(
  trials,tests,biforms,
  #patch_prolongations_old,patch_restrictions_old,
  patch_prolongations,restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter=2,mode=:preconditioner,verbose=i_am_main(parts)
);

solver = FGMRESSolver(10,gmg;verbose=i_am_main(parts),atol=1e-6)

ns = numerical_setup(symbolic_setup(solver,A),A)

x = Algebra.allocate_in_domain(A)
fill!(x,0.0)
solve!(x,ns,b)

############################################################################################
