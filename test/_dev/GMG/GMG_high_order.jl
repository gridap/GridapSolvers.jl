
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
      a(u,v) = biform(u,v,dΩ)
      local_solver = PETScLinearSolver(set_ksp_options) # IS_ConjugateGradientSolver(;reltol=1.e-6)
      patch_smoother = PatchBasedLinearSolver(a,Ph,Vh,local_solver)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.2)
    end
  end
  return smoothers
end

biform_h1(u,v,dΩ)  = ∫(v⋅u)dΩ + ∫(α*∇(v)⋅∇(u))dΩ
biform_hdiv(u,v,dΩ)  = ∫(v⋅u)dΩ + ∫(α*divergence(v)⋅divergence(u))dΩ

np       = 2    # Number of processors
D        = 2    # Problem dimension
n_refs_c = 6    # Number of refinements for the coarse model
n_levels = 2    # Number of refinement levels
order    = 0    # FE order
conf     = :HDiv  # Conformity ∈ [:H1,:HDiv]

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

reffe  = (conf==:H1) ? ReferenceFE(lagrangian,Float64,order) : ReferenceFE(raviart_thomas,Float64,order)
tests  = FESpace(mh,reffe;dirichlet_tags="boundary");
trials = TrialFESpace(tests);

α = 1.0
f = (conf==:H1) ? 1.0 : VectorValue(fill(1.0,D)...)
qdegree = order*2+1
biform  = (conf==:H1) ? biform_h1 : biform_hdiv
liform(v,dΩ)    = ∫(v⋅f)dΩ
smatrices, A, b = compute_hierarchy_matrices(trials,biform,liform,qdegree);

if conf == :H1
  smoothers = fill(RichardsonSmoother(JacobiLinearSolver(),10,9.0/8.0),n_levels-1);
else
  pbs = GridapSolvers.PatchBasedSmoothers.PatchBoundaryExclude()
  patch_decompositions = PatchDecomposition(mh;patch_boundary_style=pbs)
  patch_spaces = PatchFESpace(mh,reffe,DivConformity(),patch_decompositions,tests)
  smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)
end

restrictions, prolongations = setup_transfer_operators(trials,qdegree;mode=:residual);

GridapPETSc.with() do
  gmg = GMGLinearSolver(mh,
                        smatrices,
                        prolongations,
                        restrictions,
                        pre_smoothers=smoothers,
                        post_smoothers=smoothers,
                        coarsest_solver=PETScLinearSolver(set_ksp_options),
                        maxiter=1,
                        rtol=1.0e-10,
                        verbose=false,
                        mode=:preconditioner)

  solver = CGSolver(gmg;maxiter=100,atol=1e-10,rtol=1.e-6,verbose=i_am_main(ranks))
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = pfill(0.0,partition(axes(A,2)))
  solve!(x,ns,b)
  @time begin
    fill!(x,0.0)
    solve!(x,ns,b)
  end
  println("n_dofs = ", length(x))
end

# Result set 1: 
#   > Coarsest solver -> BackslashSolver
#   > Patch solver    -> LUSolver()
#
# Problem - np - order - ndofs - niter - time(s)
# ----------------------------------------------
#   H1       1     1     65025     3      0.57
#   H1       1     2    261121     2      1.51
#   HDiv     1     0    130560     3      7.95
#   HDiv     1     1    523264     3     40.78
# ----------------------------------------------
#   H1       2     1     65025     3      0.43
#   H1       2     2    261121     2      1.28
#   HDiv     2     0    130560     3      5.40
#   HDiv     2     1    523264     3     26.92

# Result set 2: 
#   > Coarsest solver -> MUMPS
#   > Patch solver    -> CG solver, rtol=1.e-6
#
# Problem - np - order - ndofs - niter - time(s)
# ----------------------------------------------
#   H1       1     1     65025     3     
#   H1       1     2    261121     2     
#   HDiv     1     0    130560     3     
#   HDiv     1     1    523264     3     
# ----------------------------------------------
#   H1       2     1     65025     3     
#   H1       2     2    261121     2     
#   HDiv     2     0    130560     5      9.03
#   HDiv     2     1    523264     4    153.85

# Result set 3: 
#   > Coarsest solver -> MUMPS
#   > Patch solver    -> MUMPS
#
# Problem - np - order - ndofs - niter - time(s)
# ----------------------------------------------
#   H1       1     1     65025     3     
#   H1       1     2    261121     2     
#   HDiv     1     0    130560     3     
#   HDiv     1     1    523264     3     
# ----------------------------------------------
#   H1       2     1     65025     3     
#   H1       2     2    261121     2     
#   HDiv     2     0    130560     3      9.68
#   HDiv     2     1    523264     3     41.29
