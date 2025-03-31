using PartitionedArrays
using Gridap, GridapPETSc, GridapSolvers, GridapDistributed, GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
import GridapSolvers.PatchBasedSmoothers as PBS
using Gridap.ReferenceFEs, Gridap.Geometry

function get_mesh_hierarchy(parts,Dc,np_per_level,nrefs_coarse)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (2,2)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (2,2,2)
  end
  
  num_levels   = length(np_per_level)
  cparts       = generate_subparts(parts,np_per_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,nc)
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,nrefs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

function test_solver(s,D_j)
  ns = numerical_setup(symbolic_setup(s,D_j),D_j)

  b = allocate_in_domain(D_j)
  x = allocate_in_domain(D_j)

  fill!(b,1.0)
  solve!(x,ns,b)
  err = norm(b - D_j*x)

  return err
end

function test_smoother(s,D_j)
  ns = numerical_setup(symbolic_setup(s,D_j),D_j)
  b = allocate_in_domain(D_j)
  x = allocate_in_domain(D_j)
  r = allocate_in_range(D_j)
  fill!(b,1.0)
  fill!(x,1.0)
  mul!(r,D_j,x)
  r .= b .- r
  solve!(x,ns,r)
  err = norm(b - D_j*x)
  return err
end

function get_hierarchy_matrices(mh,tests,trials,biform)
  mats = Vector{AbstractMatrix}(undef,num_levels(mh))
  A = nothing
  b = nothing
  for lev in 1:num_levels(mh)
    model = get_model(mh,lev)
    U_j = get_fe_space(trials,lev)
    V_j = get_fe_space(tests,lev)
    Ω   = Triangulation(model)
    dΩ  = Measure(Ω,2*k)
    ai(j,v_j) = biform(j,v_j,dΩ)
    if lev == 1
      Dc = num_cell_dims(model)
      f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
      li(v) = ∫(v⋅f)*dΩ
      op    = AffineFEOperator(ai,li,U_j,V_j)
      A, b  = get_matrix(op), get_vector(op)
      mats[lev] = A
    else
      mats[lev] = assemble_matrix(ai,U_j,V_j)
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
      a_j(j,v_j) = biform(j,v_j,dΩ)
      local_solver = LUSolver() # IS_ConjugateGradientSolver(;reltol=1.e-6)
      patch_smoother = PatchBasedLinearSolver(a_j,Ph,Vh,local_solver)
      smoothers[lev] = RichardsonSmoother(patch_smoother,1000,1.0)
    end
  end
  return smoothers
end

############################################################################################

np = 1
ranks = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end

# Geometry
Dc = 2
mh = get_mesh_hierarchy(ranks,Dc,[1,1],3);
model = get_model(mh,1)
println("Number of cells: ",num_cells(model))

# FESpaces
k = 1
qdegree = 2*k+2
j_bc = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
reffe_j = ReferenceFE(raviart_thomas,Float64,k)
tests  = TestFESpace(mh,reffe_j;dirichlet_tags="boundary");
trials = TrialFESpace(tests,j_bc);

biform(j,v_j,dΩ) = ∫(j⋅v_j + (∇⋅j)⋅(∇⋅v_j))*dΩ 

# Patch solver
patch_decompositions = PBS.PatchDecomposition(mh)
patch_spaces = PBS.PatchFESpace(mh,reffe_j,DivConformity(),patch_decompositions,tests);
smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)

restrictions, prolongations = setup_transfer_operators(trials,qdegree;mode=:residual);
smatrices, A, b = get_hierarchy_matrices(mh,tests,trials,biform);
println("System size: ",size(A))

gmg = GMGLinearSolver(
  smatrices,
  prolongations,
  restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  maxiter=4,
  rtol=1.0e-8,
  verbose=true,
  mode=:preconditioner
)

solver = FGMRESSolver(100,gmg;rtol=1e-6,verbose=true)

ns = numerical_setup(symbolic_setup(solver,A),A)
x = allocate_in_domain(A)
solve!(x,ns,b)


test_smoother(smoothers[1],A)


Pl = LinearSolvers.IdentitySolver()
solver2 = GMRESSolver(1000;Pl=Pl,rtol=1e-6,verbose=true)
ns2 = numerical_setup(symbolic_setup(solver2,A),A)
x2 = allocate_in_domain(A)
solve!(x2,ns2,b)
