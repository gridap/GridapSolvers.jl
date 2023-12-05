using Gridap, Gridap.Adaptivity, Gridap.ReferenceFEs
using GridapDistributed, PartitionedArrays
using GridapP4est, GridapPETSc
using GridapSolvers, GridapSolvers.MultilevelTools, GridapSolvers.LinearSolvers

function set_ksp_options(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
end

function compute_matrices(trials,tests,a::Function,l::Function,qdegree)
  nlevs = num_levels(trials)
  mh    = trials.mh

  A = nothing
  b = nothing
  mats = Vector{PSparseMatrix}(undef,nlevs)
  for lev in 1:nlevs
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model = GridapSolvers.get_model(mh,lev)
      U = get_fe_space(trials,lev)
      V = get_fe_space(tests,lev)
      Ω = Triangulation(model)
      dΩ = Measure(Ω,qdegree)
      ai(u,v) = a(u,v,dΩ)
      if lev == 1
        li(v) = l(v,dΩ)
        op    = AffineFEOperator(ai,li,U,V)
        A, b  = get_matrix(op), get_vector(op)
        mats[lev] = A
      else
        mats[lev] = assemble_matrix(ai,U,V)
      end
    end
  end
  return mats, A, b
end

function get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)  
  mh = tests.mh
  nlevs = num_levels(mh)
  smoothers = Vector{RichardsonSmoother}(undef,nlevs-1)
  for lev in 1:nlevs-1
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      PD = patch_decompositions[lev]
      Ph = get_fe_space(patch_spaces,lev)
      Vh = get_fe_space(tests,lev)
      Ω  = Triangulation(PD)
      dΩ = Measure(Ω,qdegree)
      local_solver = LUSolver() # IS_ConjugateGradientSolver(;reltol=1.e-6)
      patch_smoother = PatchBasedLinearSolver(biform,Ph,Vh,dΩ,local_solver)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.2)
    end
  end
  return smoothers
end

np       = 1    # Number of processors
D        = 3    # Problem dimension
n_refs_c = 1    # Number of refinements for the coarse model
n_levels = 2    # Number of refinement levels
order    = 1    # FE order

ranks = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end

domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
nc = Tuple(fill(2,D))
cmodel = CartesianDiscreteModel(domain,nc)

mh = GridapP4est.with(ranks) do
  num_parts_x_level = fill(np,n_levels)
  coarse_model = OctreeDistributedDiscreteModel(ranks,cmodel,n_refs_c)
  return ModelHierarchy(ranks,coarse_model,num_parts_x_level)
end;
n_cells = num_cells(GridapSolvers.get_model(mh,1))

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
reffe_j = ReferenceFE(raviart_thomas,Float64,order-1)

tests_u  = FESpace(mh,reffe_u;dirichlet_tags="boundary");
trials_u = TrialFESpace(tests_u);
tests_j  = FESpace(mh,reffe_j;dirichlet_tags="boundary");
trials_j = TrialFESpace(tests_j);

trials = MultiFieldFESpace([trials_u,trials_j]);
tests  = MultiFieldFESpace([tests_u,tests_j]);

β = 1.0
γ = 1.0
B = VectorValue(0.0,0.0,1.0)
f = VectorValue(fill(1.0,D)...)
qdegree = order*2+1
biform((u,j),(v_u,v_j),dΩ) = ∫(β*∇(u)⊙∇(v_u) -γ*(j×B)⋅v_u + j⋅v_j - (u×B)⋅v_j)dΩ
liform((v_u,v_j),dΩ) = ∫(v_u⋅f)dΩ
smatrices, A, b = compute_matrices(trials,tests,biform,liform,qdegree);

pbs = GridapSolvers.PatchBasedSmoothers.PatchBoundaryExclude()
patch_decompositions = PatchDecomposition(mh;patch_boundary_style=pbs)
patch_spaces = PatchFESpace(tests,patch_decompositions);
smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)

smoother_ns = numerical_setup(symbolic_setup(smoothers[1],A),A)

restrictions, prolongations = setup_transfer_operators(trials,qdegree;mode=:residual);


#GridapPETSc.with() do
#  gmg = GMGLinearSolver(mh,
#                        smatrices,
#                        prolongations,
#                        restrictions,
#                        pre_smoothers=smoothers,
#                        post_smoothers=smoothers,
#                        coarsest_solver=PETScLinearSolver(set_ksp_options),
#                        maxiter=1,
#                        rtol=1.0e-10,
#                        verbose=false,
#                        mode=:preconditioner)
#
#  solver = CGSolver(gmg;maxiter=100,atol=1e-10,rtol=1.e-6,verbose=i_am_main(ranks))
#  ns = numerical_setup(symbolic_setup(solver,A),A)
#
#  x = pfill(0.0,partition(axes(A,2)))
#  solve!(x,ns,b)
#  @time begin
#    fill!(x,0.0)
#    solve!(x,ns,b)
#  end
#  println("n_dofs = ", length(x))
#end