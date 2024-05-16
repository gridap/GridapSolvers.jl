using FileIO
using Gridap, Gridap.Algebra, Gridap.MultiField
using PartitionedArrays, GridapDistributed, GridapSolvers, GridapPETSc
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.BlockSolvers

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

function driver(parts,np_per_level,nc)
  t = PTimer(parts;verbose=true)

  tic!(t;barrier=true)
  # Geometry
  mh = CartesianModelHierarchy(parts,np_per_level,(0,1,0,1),nc;add_labels!)
  model = get_model(mh,1)

  # FE spaces
  fe_order = 2
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
  biforms = map(mhl -> get_bilinear_form(mhl,biform_u,qdegree),mh)
  smoothers = map(mhl -> RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0), view(mh,1:num_levels(mh)-1))
  restrictions, prolongations = setup_transfer_operators(
    trials_u, qdegree; mode=:residual, solver=CGSolver(JacobiLinearSolver())
  )
  solver_u = GMGLinearSolver(
    mh,trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=2,mode=:preconditioner,verbose=i_am_main(parts)
  )
  solver_u.log.depth = 4

  # PCG solver for the pressure block
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  solver_p.log.depth = 4

  # Block triangular preconditioner
  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  P = BlockTriangularSolver(blocks,[solver_u,solver_p])
  solver = FGMRESSolver(10,P;rtol=1.e-8,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)
  toc!(t,"Setup")

  tic!(t;barrier=true)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  toc!(t,"Solver")

  # Postprocess
  ncells  = num_cells(model)
  ndofs_u = num_free_dofs(U)
  ndofs_p = num_free_dofs(Q)
  output = Dict{String,Any}()
  map_main(t.data) do timer_data
    output["np"] = np_per_level[1]
    output["nc"] = nc
    output["np_per_level"] = np_per_level
    output["ncells"]  = ncells
    output["ndofs_u"] = ndofs_u
    output["ndofs_p"] = ndofs_p
    output["niter"]   = solver.log.num_iters
    merge!(output,timer_data)
  end
  return output
end

function main(;
  nr = 1,
  np = 1,
  np_per_level = [np,1],
  nc = (10,10),
  petsc_options = "-ksp_monitor -ksp_error_if_not_converged true -ksp_converged_reason",
  title = "data/stokes_np_$(np)_nc_$(prod(nc))"
)
  parts = with_mpi() do distribute
    distribute(LinearIndices((prod(np),)))
  end
  GridapPETSc.with(;args=split(petsc_options)) do
    driver(parts,np_per_level,nc)
    for ir in 1:nr
      if i_am_main(parts)
        println(repeat('-',28))
        println(" ------- ITERATION $(ir) ------- ")
        println(repeat('-',28))
      end
      title_ir = "$(title)_$(ir)"
      output = driver(parts,np_per_level,nc)
      map_main(parts) do p
        output["ir"] = ir
        save("$title_ir.bson",output)
      end
    end
  end
end
