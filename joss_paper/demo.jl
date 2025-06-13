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
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
end

with_mpi() do distribute
  np_per_level = [(2,2),(1,1)] # Number of processors per GMG level
  parts = distribute(LinearIndices((prod(np_per_level[1]),)))

  # Create multi-level mesh
  domain = (0,1,0,1) # Cartesian domain (xmin,xmax,ymin,ymax)
  ncells = (10,10)   # Number of cells
  mh = CartesianModelHierarchy(parts,np_per_level,domain,ncells;add_labels!)
  model = get_model(mh,1) # Finest mesh

  # Create FESpaces
  fe_order = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},fe_order)
  reffe_p = ReferenceFE(lagrangian,Float64,fe_order-1;space=:P)

  tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["walls","top"])
  trials_u = TrialFESpace(tests_u,[VectorValue(0.0,0.0),VectorValue(1.0,0.0)])
  U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

  mfs = BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  # Weak formulation
  f = VectorValue(1.0,1.0)
  biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ
  biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫((∇⋅v)*p)dΩ - ∫((∇⋅u)*q)dΩ
  liform((v,q),dΩ) = ∫(v⋅f)dΩ

  # Assemble linear system
  qdegree = 2*(fe_order+1) # Quadrature degree
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
    trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=4,mode=:solver
  )

  # PCG solver for the pressure block
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6)

  # 2x2 Block triangular preconditioner
  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  P = BlockTriangularSolver(blocks,[solver_u,solver_p])

  # Global solver
  solver = FGMRESSolver(10,P;rtol=1.e-8,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)

  # Solve
  x = allocate_in_domain(A)
  fill!(x,0.0)
  solve!(x,ns,b)

  # Postprocess
  uh, ph = FEFunction(X,x)
  writevtk(Ω,"demo",cellfields=["uh"=>uh,"ph"=>ph])
end