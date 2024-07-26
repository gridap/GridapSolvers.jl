
using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers

using Gridap.MultiField, Gridap.Arrays, Gridap.Geometry

using GridapSolvers.PatchBasedSmoothers

function add_labels!(labels)
  add_tag_from_tags!(labels,"noslip",[1:20...,23,24,25,26])
  add_tag_from_tags!(labels,"insulating",[1:20...,25,26])
  add_tag_from_tags!(labels,"topbottom",[21,22])
end

ranks = with_debug() do distribute
  distribute(LinearIndices((4,)))
end

nrefs = (2,2,2)
isperiodic = (false,false,true)

np_per_level = [(2,2,1),(2,2,1)]

mh = CartesianModelHierarchy(
  ranks,
  np_per_level,
  (-1.,1.,-1.,1.,0.,0.1),
  (4,4,4);
  nrefs,
  isperiodic,
  add_labels!,
)

order = 2
qdegree = 2*order
reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
reffe_j = ReferenceFE(raviart_thomas,Float64,order-1)
tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["noslip"]);
tests_j = TestFESpace(mh,reffe_j,dirichlet_tags=["insulating"]);
tests = MultiFieldFESpace([tests_u,tests_j])

PD = PatchDecomposition(mh)
Ph = PatchFESpace(tests,PD)

biform((u,j),(v,s),dΩ) = ∫(u⋅v + j⋅s)dΩ

nlevs = num_levels(mh)
smoothers = map(view(tests,1:nlevs-1),PD,Ph) do tests, PD, Ph
  Vh = get_fe_space(tests)
  Ω  = Triangulation(PD)
  dΩ = Measure(Ω,qdegree)
  ap(u,v) = biform(u,v,dΩ)
  patch_solver = PatchBasedLinearSolver(ap,Ph,Vh)
  RichardsonSmoother(patch_solver,10,0.2)
end

smatrices = map(view(mh,1:nlevs-1),view(tests,1:nlevs-1)) do mhlev,tests
  Vh = get_fe_space(tests)
  model = get_model(mhlev)
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  a(u,v) = biform(u,v,dΩ)
  assemble_matrix(a,Vh,Vh)
end

smoother_ns = map(smoothers,smatrices) do s, m
  numerical_setup(symbolic_setup(s,m),m)
end

prolongations = map(view(linear_indices(mh),1:nlevs-1),view(mh,1:nlevs-1),PD) do lev,mhlev,PD
  model = get_model(mhlev)
  Ω  = Triangulation(PD)
  dΩ = Measure(Ω,qdegree)
  lhs(u,v) = biform(u,v,dΩ)
  rhs(u,v) = biform(u,v,dΩ)
  PatchProlongationOperator(lev,tests,PD,lhs,rhs)
end
