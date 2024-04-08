using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function get_patch_smoothers(mh,tests,biform,patch_decompositions,qdegree)
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),patch_decompositions,patch_spaces) do tests, PD, Ph
    Vh = get_fe_space(tests)
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

nc = (8,8)
fe_order = 2

np = 4
np_per_level = [np,1]

with_mpi() do distribute
  parts = distribute(LinearIndices((prod(np),)))

  # Coarse geometry
  cmodel = CartesianDiscreteModel((0,1,0,1),nc)
  labels = get_face_labeling(cmodel)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"walls",[1,5,7])
  add_tag_from_tags!(labels,"right",[2,8])

  # Mesh refinement using GridapP4est
  cparts = generate_subparts(parts,np_per_level[end])
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,0)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  model = get_model(mh,1)

  # FE spaces
  qdegree = 2*(fe_order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},fe_order)
  reffe_p = ReferenceFE(lagrangian,Float64,fe_order-1;space=:P)

  tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["walls","top"]);
  trials_u = TrialFESpace(tests_u,[VectorValue(0.0,0.0),VectorValue(1.0,0.0)]);
  U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
  Q = TestFESpace(model,reffe_p;conformity=:L2) 

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  # Weak formulation
  α = 1.e2
  f = VectorValue(1.0,1.0)
  Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,fe_order-1;quad_order=qdegree,space=:P)
  graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
  biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ + graddiv(u,v,dΩ)
  biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
  liform((v,q),dΩ) = ∫(v⋅f)dΩ

  # Finest level
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  a(u,v) = biform(u,v,dΩ)
  l(v) = liform(v,dΩ)
  op = AffineFEOperator(a,l,X,Y)
  A, b = get_matrix(op), get_vector(op);

  # GMG preconditioner for the velocity block
  biforms = map(mhl -> get_bilinear_form(mhl,biform_u,qdegree),mh)
  patch_decompositions = PatchDecomposition(mh)
  smoothers = get_patch_smoothers(
    mh,tests_u,biform_u,patch_decompositions,qdegree
  )
  restrictions = setup_restriction_operators(
    tests_u,qdegree;mode=:residual,solver=IS_ConjugateGradientSolver(;reltol=1.e-6)
  )
  prolongations = setup_patch_prolongation_operators(
    tests_u,biform_u,graddiv,qdegree
  )
  solver_u = GMGLinearSolver(
    mh,trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=2,mode=:preconditioner,verbose=i_am_main(parts)
  )

  # PCG solver for the pressure block
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))

  # Block triangular preconditioner
  blocks = [LinearSystemBlock(), LinearSystemBlock();
            LinearSystemBlock(), BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
  coeffs = [1.0 1.0;
            0.0 1.0]  
  P = BlockTriangularSolver(blocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts))
  
  ns = numerical_setup(symbolic_setup(solver,A),A)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  uh, ph = FEFunction(X,x)
  writevtk(Ω,"results",cellfields=["uh"=>uh,"ph"=>ph])
end
