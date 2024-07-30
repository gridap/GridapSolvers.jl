using Gridap, Gridap.Algebra, Gridap.MultiField
using PartitionedArrays, GridapDistributed, GridapSolvers

using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers

function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

function add_labels!(labels)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"bottom",[1,2,5])
end

np = (2,2)
np_per_level = [np,(1,1)]
nc = (10,10)
fe_order = 2

with_mpi() do distribute
  parts = distribute(LinearIndices((prod(np),)))

  # Geometry
  mh = CartesianModelHierarchy(parts,np_per_level,(0,1,0,1),nc;add_labels!)
  model = get_model(mh,1)

  # FE spaces
  qdegree = 2*(fe_order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},fe_order)
  reffe_p = ReferenceFE(lagrangian,Float64,fe_order-1;space=:P)

  tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["bottom","top"])
  trials_u = TrialFESpace(tests_u,[VectorValue(0.0,0.0),VectorValue(1.0,0.0)])
  U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
  Q = TestFESpace(model,reffe_p;conformity=:L2) 

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  # Weak formulation
  f = VectorValue(1.0,1.0)
  biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ
  biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫((∇⋅v)*p)dΩ - ∫((∇⋅u)*q)dΩ
  liform((v,q),dΩ) = ∫(v⋅f)dΩ

  # Finest level
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  op = AffineFEOperator((u,v)->biform(u,v,dΩ),v->liform(v,dΩ),X,Y)
  A, b = get_matrix(op), get_vector(op)

  # GMG preconditioner for the velocity block
  biforms = map(mh) do mhl
    get_bilinear_form(mhl,biform_u,qdegree)
  end
  smoothers = map(view(mh,1:num_levels(mh)-1)) do mhl
    RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0)
  end
  restrictions, prolongations = setup_transfer_operators(
    trials_u, qdegree; mode=:residual, solver=CGSolver(JacobiLinearSolver())
  )
  solver_u = GMGLinearSolver(
    mh,trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=2,mode=:preconditioner
  )

  # PCG solver for the pressure block
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)

  # Block triangular preconditioner
  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  P = BlockTriangularSolver(blocks,[solver_u,solver_p])
  solver = GMRESSolver(10;Pr=P,rtol=1.e-8,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = allocate_in_domain(A)
  fill!(x,0.0)
  solve!(x,ns,b)
  uh, ph = FEFunction(X,x)
  writevtk(Ω,"demo",cellfields=["uh"=>uh,"ph"=>ph])
end