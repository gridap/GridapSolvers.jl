
function stokes_gmg_driver(parts,mh)
  t = PTimer(parts;verbose=true)

  tic!(t;barrier=true)
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
  toc!(t,"FESpaces")

  # Weak formulation
  tic!(t;barrier=true)
  f = VectorValue(1.0,1.0)
  biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ
  biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫((∇⋅v)*p)dΩ - ∫((∇⋅u)*q)dΩ
  liform((v,q),dΩ) = ∫(v⋅f)dΩ

  # Finest level
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  op = AffineFEOperator((u,v)->biform(u,v,dΩ),v->liform(v,dΩ),X,Y)
  A, b = get_matrix(op), get_vector(op)
  toc!(t,"Integration")

  tic!(t;barrier=true)
  # GMG preconditioner for the velocity block
  biforms = map(mhl -> get_bilinear_form(mhl,biform_u,qdegree),mh)
  smoothers = map(mhl -> RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0), view(mh,1:num_levels(mh)-1))
  restrictions, prolongations = setup_transfer_operators(
    trials_u, qdegree; mode=:residual, solver=CGSolver(JacobiLinearSolver(),verbose=false)
  )
  solver_u = GMGLinearSolver(
    mh,trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=PETScLinearSolver(petsc_mumps_setup),
    maxiter=3,mode=:preconditioner,verbose=i_am_main(parts)
  )
  solver_u.log.depth = 4

  # PCG solver for the pressure block
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-10,verbose=i_am_main(parts))
  solver_p.log.depth = 4

  # Block triangular preconditioner
  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  P = BlockTriangularSolver(blocks,[solver_u,solver_p])
  solver = FGMRESSolver(15,P;rtol=1.e-8,maxiter=20,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)
  toc!(t,"SolverSetup")

  tic!(t;barrier=true)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)
  toc!(t,"Solver")

  # Cleanup PETSc C-allocated objects
  nlevs = num_levels(mh)
  GridapSolvers.MultilevelTools.with_level(mh,nlevs) do mhl
    P_ns   = ns.Pr_ns
    gmg_ns = P_ns.block_ns[1].ns
    finalize(gmg_ns.coarsest_solver_cache.X)
    finalize(gmg_ns.coarsest_solver_cache.B)
    finalize(gmg_ns.coarsest_solver_cache)
    GridapPETSc.gridap_petsc_gc()
  end

  # Postprocess
  ncells  = num_cells(model)
  ndofs_u = num_free_dofs(U)
  ndofs_p = num_free_dofs(Q)
  output = Dict{String,Any}()
  map_main(t.data) do timer_data
    output["ncells"]  = ncells
    output["ndofs_u"] = ndofs_u
    output["ndofs_p"] = ndofs_p
    output["niter"]   = solver.log.num_iters
    merge!(output,timer_data)
  end
  return output
end

function stokes_gmg_main(;
  nr = 1,
  np = (1,1),
  np_per_level = [np,np],
  nc = (10,10),
  petsc_options = "-ksp_monitor -ksp_error_if_not_converged true -ksp_converged_reason",
  title = "data/stokes_np_$(prod(np))_nc_$(prod(nc))",
  mesher = :gridap,
  mode = :mpi
)
  @assert mesher ∈ [:gridap,:p4est]
  @assert mode ∈ [:mpi,:debug]
  with_mode = (mode == :mpi) ? with_mpi : with_debug
  parts = with_mode() do distribute
    distribute(LinearIndices((prod(np),)))
  end

  t = PTimer(parts;verbose=true)
  tic!(t;barrier=true)
  if mesher == :gridap
    mh = CartesianModelHierarchy(parts,np_per_level,(0,1,0,1),nc;add_labels!)
  elseif mesher == :p4est
    np_per_level = map(prod,np_per_level)
    mh = P4estCartesianModelHierarchy(parts,np_per_level,(0,1,0,1),nc;add_labels!)
  end
  toc!(t,"Geometry")

  GridapPETSc.with(;args=split(petsc_options)) do
    stokes_gmg_driver(parts,mh)
    for ir in 1:nr
      map_main(parts) do p
        println(repeat('-',28))
        println(" ------- ITERATION $(ir) ------- ")
        println(repeat('-',28))
      end
      output = stokes_gmg_driver(parts,mh)
      map_main(parts) do p
        output["ir"] = ir
        output["np"] = np
        output["nc"] = nc
        output["nl"] = length(np_per_level)
        output["np_per_level"] = np_per_level
        save("$(title)_$(ir).bson",output)
      end
      GridapPETSc.gridap_petsc_gc()
    end
  end
end
