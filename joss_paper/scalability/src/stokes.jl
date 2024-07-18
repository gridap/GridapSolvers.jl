
function stokes_driver(parts,model)
  t = PTimer(parts;verbose=true)

  tic!(t;barrier=true)

  # FE spaces
  fe_order = 2
  qdegree = 2*(fe_order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},fe_order)
  reffe_p = ReferenceFE(lagrangian,Float64,fe_order-1;space=:P)

  V = TestFESpace(model,reffe_u,dirichlet_tags=["bottom","top"])
  U = TrialFESpace(V,[VectorValue(0.0,0.0),VectorValue(1.0,0.0)])
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

  solver_u = ASMSolver()
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-10,verbose=i_am_main(parts))
  solver_p.log.depth = 4

  # Block triangular preconditioner
  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  P = BlockTriangularSolver(blocks,[solver_u,solver_p])
  solver = FGMRESSolver(15,P;rtol=1.e-8,verbose=i_am_main(parts))
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
    output["ncells"]  = ncells
    output["ndofs_u"] = ndofs_u
    output["ndofs_p"] = ndofs_p
    output["niter"]   = solver.log.num_iters
    merge!(output,timer_data)
  end
  return output
end

function stokes_main(;
  nr = 1,
  np = (1,1),
  nc = (10,10),
  petsc_options = "-ksp_monitor -ksp_error_if_not_converged true -ksp_converged_reason",
  title = "data/stokes_np_$(np)_nc_$(prod(nc))",
)
  parts = with_mpi() do distribute
    distribute(LinearIndices((prod(np),)))
  end

  model = CartesianDiscreteModel(parts,np,(0,1,0,1),nc)
  map(add_labels!,local_views(get_face_labeling(model)))

  GridapPETSc.with(;args=split(petsc_options)) do
    stokes_driver(parts,model)
    for ir in 1:nr
      map_main(parts) do p
        println(repeat('-',28))
        println(" ------- ITERATION $(ir) ------- ")
        println(repeat('-',28))
      end
      output = stokes_driver(parts,model)
      map_main(parts) do p
        output["ir"] = ir
        output["np"] = np
        output["nc"] = nc
        save("$(title)_$(ir).bson",output)
      end
      GridapPETSc.gridap_petsc_gc()
    end
  end
end