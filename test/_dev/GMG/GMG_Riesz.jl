
using LinearAlgebra
using FillArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra, Gridap.Adaptivity
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers

function get_patch_smoothers(mh,tests,biform,qdegree)
  nlevs = num_levels(mh)
  smoothers = map(view(mh,1:nlevs-1),view(tests,1:nlevs-1)) do mhl,tests
    model = get_model(mhl)
    Vh = get_fe_space(tests)
    ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
    Ω  = Geometry.PatchTriangulation(model,ptopo)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    solver = PatchBasedSmoothers.PatchSolver(ptopo,Vh,Vh,ap;mask=:boundary,collect_factorizations=false)
    return RichardsonSmoother(solver,10,0.2)
  end
  return smoothers
end

function get_patch_smoothers_old(mh,tests,biform,qdegree)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),view(mh,1:nlevs-1)) do tests, mhl
    model = get_model(mhl)
    Vh = get_fe_space(tests)
    PD = PatchDecomposition(model)
    Ph = PatchFESpace(Vh,PD)
    Ω  = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh)
    return RichardsonSmoother(patch_smoother,10,0.2)
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

order = 1
qdegree = 2*order

reffe_h1 = ReferenceFE(raviart_thomas,VectorValue{Dc,Float64},order)
reffe_hdiv = ReferenceFE(raviart_thomas,Float64,order-1)
reffe_hcurl = ReferenceFE(nedelec,Float64,order-1)

α = 1.e0
biform_h1(u,v,dΩ) = ∫(∇(u)⊙∇(v) + α * divergence(u)⋅divergence(v))dΩ
biform_hdiv(u,v,dΩ) = ∫(u⋅v + α * divergence(u)⋅divergence(v))dΩ
biform_hcurl(u,v,dΩ) = ∫(u⋅v + α * curl(u)⋅curl(v))dΩ

u_h1(x) = ifelse(Dc==2, VectorValue(x[1]+x[2],-x[2]) , VectorValue(x[1]+x[2],-x[2],0.0))
u_hdiv(x) = ifelse(Dc==2, VectorValue(x[1]+x[2],-x[2]) , VectorValue(x[1]+x[2],-x[2],0.0))
u_hcurl(x) = ifelse(Dc==2, VectorValue(x[1]+x[2],-x[2]) , VectorValue(x[1]+x[2],-x[2],0.0))

conformity = :Hdiv # :Hdiv, :Hcurl
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

tests  = TestFESpace(mh,reffe,dirichlet_tags=["boundary"]);
trials = TrialFESpace(tests,[u_exact]);
U, V = get_fe_space(trials,1), get_fe_space(tests,1)

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)
a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,U,V)
A, b = get_matrix(op), get_vector(op);

biforms = map(mhl -> get_bilinear_form(mhl,biform,qdegree), mh)

smoothers = get_patch_smoothers(
  mh,tests,biform,qdegree
)
smoothers_old = get_patch_smoothers_old(
  mh,tests,biform,qdegree
)
prolongations = setup_prolongation_operators(
  tests,qdegree;mode=:residual
)
restrictions = setup_restriction_operators(
  tests,qdegree;mode=:residual
)

gmg = GMGLinearSolver(
  trials,tests,biforms,
  prolongations,restrictions,
  pre_smoothers=smoothers_old,
  post_smoothers=smoothers_old,
  coarsest_solver=LUSolver(),
  maxiter=3,mode=:preconditioner,verbose=i_am_main(parts)
)

solver = FGMRESSolver(10,gmg;verbose=i_am_main(parts),atol=1e-6)

ns = numerical_setup(symbolic_setup(solver,A),A)

x = Algebra.allocate_in_domain(A)
fill!(x,0.0)
solve!(x,ns,b)

############################################################################################

prows = ns.Pr_ns.pre_smoothers_caches[1].Mns.patch_rows
pmats = ns.Pr_ns.pre_smoothers_caches[1].Mns.solver.patch_mats
pfact = ns.Pr_ns.pre_smoothers_caches[1].Mns.patch_factorizations

S = smoothers[1].M
Sns = numerical_setup(symbolic_setup(S,A),A)
x = Algebra.allocate_in_domain(A)
fill!(x,0.0)
r = A*x - b
solve!(x,Sns,r)


PS = smoothers_old[1].M
PV = PS.patch_space
PD = PatchDecomposition(model)
PSns = numerical_setup(symbolic_setup(PS,A),A)
y = Algebra.allocate_in_domain(A)
fill!(y,0.0)
r2 = A*y - b
solve!(y,PSns,r2)

norm(x-y)

ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
prows = Sns.patch_cols

pcells = map(Geometry.get_patch_cells,local_views(ptopo))
pcell_old = PD.patch_cells
pcells == pcell_old

pmats = Sns.solver.patch_mats
mat = PSns.local_A

dof_to_pdof = map(s -> s.dof_to_pdof, PV.spaces)
map(s -> s.patch_cell_dofs_ids, PV.spaces)

mref = pmats.maps[1].cell_data[1][1][1]
pcell_to_pdof = pmats.maps[1].cell_data[3][1]

ptrian = Geometry.PatchTriangulation(model,ptopo)
get_cell_dof_ids(V,ptrian)

