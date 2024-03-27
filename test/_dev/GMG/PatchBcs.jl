
using MPI
using Test
using LinearAlgebra
using IterativeSolvers
using FillArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers


function get_patch_smoothers(tests,patch_decompositions,biform,qdegree)
  mh = tests.mh
  patch_spaces = PatchFESpace(tests,patch_decompositions)
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
      ap(u,v) = biform(u,v,dΩ)
      patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.1)
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
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

u(x) = VectorValue(x[1]^2,-2*x[1]*x[2])
f(x) = VectorValue(x[1],x[2])

np = 1
nc = (6,6)
np_per_level = [np,np]
parts = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end
mh = get_mesh_hierarchy(parts,nc,np_per_level)

α = 1000.0
#biform(u,v,dΩ)  = ∫(v⋅u)dΩ + ∫(α*divergence(v)⋅divergence(u))dΩ
biform(u,v,dΩ)  = ∫(∇(v)⊙∇(u))dΩ + ∫(α*divergence(v)⋅divergence(u))dΩ
liform(v,dΩ)    = ∫(v⋅f)dΩ

order = 2
qdegree = 2*(order+1)
#reffe = ReferenceFE(raviart_thomas,Float64,order-1)
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

tests  = TestFESpace(mh,reffe,dirichlet_tags="boundary");
trials = TrialFESpace(tests,u);

patch_decompositions = PatchDecomposition(mh)
smoothers = get_patch_smoothers(tests,patch_decompositions,biform,qdegree)

smatrices, A, b = compute_hierarchy_matrices(trials,tests,biform,liform,qdegree);

coarse_solver = LUSolver()
restrictions, prolongations = setup_transfer_operators(trials,
                                                        qdegree;
                                                        mode=:residual,
                                                        solver=LUSolver());
gmg = GMGLinearSolver(mh,
                      smatrices,
                      prolongations,
                      restrictions,
                      pre_smoothers=smoothers,
                      post_smoothers=smoothers,
                      coarsest_solver=coarse_solver,
                      maxiter=2,
                      rtol=1.0e-8,
                      verbose=true,
                      mode=:preconditioner)
gmg.log.depth += 1

solver = CGSolver(gmg;maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
ns = numerical_setup(symbolic_setup(solver,A),A)

# Solve
x = pfill(0.0,partition(axes(A,2)))
solve!(x,ns,b)
