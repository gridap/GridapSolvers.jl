using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

function get_hierarchy_matrices(trials,tests,biform,qdegree)
  nlevs = num_levels(trials)
  mh    = trials.mh

  mats = Vector{PSparseMatrix}(undef,nlevs)
  for lev in 1:nlevs
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model = MultilevelTools.get_model(mh,lev)
      U = GridapSolvers.MultilevelTools.get_fe_space(trials,lev)
      V = GridapSolvers.MultilevelTools.get_fe_space(tests,lev)
      Ω = Triangulation(model)
      dΩ = Measure(Ω,qdegree)
      ai(u,v) = biform(u,v,dΩ)
      mats[lev] = assemble_matrix(ai,U,V)
    end
  end
  return mats
end

function get_patch_smoothers(tests,patch_decompositions,biform,qdegree)
  mh = tests.mh
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  nlevs = num_levels(mh)
  smoothers = Vector{RichardsonSmoother}(undef,nlevs-1)
  for lev in 1:nlevs-1
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      PD = patch_decompositions[lev]
      Ph = GridapSolvers.MultilevelTools.get_fe_space(patch_spaces,lev)
      Vh = GridapSolvers.MultilevelTools.get_fe_space(tests,lev)
      Ω  = Triangulation(PD)
      dΩ = Measure(Ω,qdegree)
      ap(u,v) = biform(u,v,dΩ)
      patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.2)
    end
  end
  return smoothers
end

function get_mesh_hierarchy(parts,nc,np_per_level)
  Dc = length(nc)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  num_refs_coarse = (Dc == 2) ? 1 : 0
  
  num_levels   = length(np_per_level)
  cparts       = generate_subparts(parts,np_per_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,nc)

  labels = get_face_labeling(cmodel)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"walls",[1,2,5,7,8])

  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

np = 1
nc = (8,8)
np_per_level = [np,np]
parts = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end
mh = get_mesh_hierarchy(parts,nc,np_per_level);
model = MultilevelTools.get_model(mh,1)

order = 2
qdegree = 2*(order+1)
Dc = length(nc)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["walls","top"]);
trials_u = TrialFESpace(tests_u,[VectorValue(0.0,0.0),VectorValue(1.0,0.0)]);

U = GridapSolvers.MultilevelTools.get_fe_space(trials_u,1)
V = GridapSolvers.MultilevelTools.get_fe_space(tests_u,1)
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

α = 1.e6
f = VectorValue(1.0,1.0)
Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ + graddiv(u,v,dΩ)
biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
liform((v,q),dΩ) = ∫(v⋅f)dΩ

patch_decompositions = PatchDecomposition(mh)
smoothers = get_patch_smoothers(tests_u,patch_decompositions,biform_u,qdegree)
smatrices = get_hierarchy_matrices(trials_u,tests_u,biform_u,qdegree);

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)
a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,X,Y)
A, b = get_matrix(op), get_vector(op);
Auu = blocks(A)[1,1]

restrictions = setup_restriction_operators(tests_u,qdegree;mode=:residual,solver=LUSolver());
prolongations = setup_patch_prolongation_operators(tests_u,patch_decompositions,biform_u,graddiv,qdegree);

gmg = GMGLinearSolver(mh,
                      smatrices,
                      prolongations,
                      restrictions,
                      pre_smoothers=smoothers,
                      post_smoothers=smoothers,
                      coarsest_solver=LUSolver(),
                      maxiter=4,
                      rtol=1.0e-8,
                      verbose=true,
                      mode=:preconditioner)
gmg.log.depth += 1

solver_u = FGMRESSolver(5,gmg;maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
ns_u = numerical_setup(symbolic_setup(solver_u,Auu),Auu)

x_u = pfill(0.0,partition(axes(Auu,2)))
b_u = blocks(b)[1]
solve!(x_u,ns_u,b_u)

# Solve

solver_p = CGSolver(RichardsonSmoother(JacobiLinearSolver(),10,0.2);maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)

using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver
diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]  
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
ns = numerical_setup(symbolic_setup(solver,A),A)

x = Gridap.Algebra.allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)

# Postprocess

model = get_model(mh,1)
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

U = get_fe_space(trials,1)
uh = FEFunction(U,x)

uh_exact = interpolate(u_exact,U)
eh = uh - uh_exact
E = sqrt(sum(∫(eh⋅eh)dΩ))
