
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

function get_mesh_hierarchy(parts,cmodel,num_refs_coarse,np_per_level)
  num_levels   = length(np_per_level)
  cparts       = generate_subparts(parts,np_per_level[num_levels])
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

function get_hierarchy_matrices_old(
  trials::FESpaceHierarchy,
  tests::FESpaceHierarchy,
  a::Function,
  l::Function,
  qdegree::Integer;
  is_nonlinear::Bool=false
)
  nlevs = num_levels(trials)
  mh    = trials.mh

  A = nothing
  b = nothing
  mats = Vector{PSparseMatrix}(undef,nlevs)
  for lev in 1:nlevs
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model = get_model(mh,lev)
      U = get_fe_space(trials,lev)
      V = get_fe_space(tests,lev)
      Ω = Triangulation(model)
      dΩ = Measure(Ω,qdegree)
      ai(u,v) = is_nonlinear ? a(zero(U),u,v,dΩ) : a(u,v,dΩ)
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

function get_hierarchy_matrices(
  trials::FESpaceHierarchy,
  tests::FESpaceHierarchy,
  a::Function,
  qdegree::Integer;
  is_nonlinear::Bool=false
)
  nlevs = num_levels(trials)
  mh    = trials.mh

  mats = Vector{PSparseMatrix}(undef,nlevs)
  for lev in 1:nlevs
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model = get_model(mh,lev)
      U = get_fe_space(trials,lev)
      V = get_fe_space(tests,lev)
      Ω = Triangulation(model)
      dΩ = Measure(Ω,qdegree)
      ai(u,v) = is_nonlinear ? a(zero(U),u,v,dΩ) : a(u,v,dΩ)
      mats[lev] = assemble_matrix(ai,U,V)
    end
  end
  return mats
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
      ap(u,du,v) = biform(u,du,v,dΩ)
      patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh;is_nonlinear=true)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.2)
    end
  end
  return smoothers
end

function add_hunt_tags!(model)
  labels = get_face_labeling(model)
  tags_u = append!(collect(1:20),[23,24,25,26])
  tags_j = append!(collect(1:20),[25,26])
  add_tag_from_tags!(labels,"noslip",tags_u)
  add_tag_from_tags!(labels,"insulating",tags_j)
end

##########################

Dc = 3
np = 1
nc = (4,4,3)
parts = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end
domain = (0.0,1.0,0.0,1.0,0.0,1.0)
cmodel = CartesianDiscreteModel(domain,nc;isperiodic=(false,false,true))
add_hunt_tags!(cmodel)
mh = get_mesh_hierarchy(parts,cmodel,0,[1,1]);

order = 2
reffe_u  = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
tests_u  = FESpace(mh,reffe_u;dirichlet_tags="noslip");
trials_u = TrialFESpace(tests_u);

reffe_j  = ReferenceFE(raviart_thomas,Float64,order-1)
tests_j  = FESpace(mh,reffe_j;dirichlet_tags="insulating");
trials_j = TrialFESpace(tests_j);

trials = MultiFieldFESpace([trials_u,trials_j]);
tests  = MultiFieldFESpace([tests_u,tests_j]);
spaces = tests, trials

α = 1.0
β = 1.0
γ = 10000.0
B = VectorValue(0.0,1.0,0.0)
f = VectorValue(0.0,0.0,1.0)
η_u, η_j = 10.0,10.0

conv(u,∇u) = (∇u')⋅u
a_al((u,j),(v_u,v_j),dΩ) = ∫(η_u*(∇⋅u)⋅(∇⋅v_u))*dΩ + ∫(η_j*(∇⋅j)⋅(∇⋅v_j))*dΩ
a_mhd((u,j),(v_u,v_j),dΩ) = ∫(β*∇(u)⊙∇(v_u) -γ*(j×B)⋅v_u + j⋅v_j - (u×B)⋅v_j)dΩ
c_mhd((u,j),(v_u,v_j),dΩ) = ∫( α*v_u⋅(conv∘(u,∇(u))) ) * dΩ
dc_mhd((u,j),(du,dj),(v_u,v_j),dΩ) = ∫(α*v_u⋅( (conv∘(u,∇(du))) + (conv∘(du,∇(u)))))dΩ
rhs((u,j),(v_u,v_j),dΩ) = ∫(f⋅v_u)dΩ

jac(x0,x,y,dΩ) = a_mhd(x,y,dΩ) + a_al(x,y,dΩ) + dc_mhd(x0,x,y,dΩ)
res(x0,y,dΩ) = a_mhd(x0,y,dΩ) + a_al(x0,y,dΩ) + c_mhd(x0,y,dΩ) - rhs(x0,y,dΩ)


qdegree = 2*(order+1)
patch_decompositions = PatchDecomposition(mh)
patch_spaces = PatchFESpace(tests,patch_decompositions);
smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,jac,qdegree)

smatrices = get_hierarchy_matrices(trials,tests,jac,qdegree;is_nonlinear=true);
A = smatrices[1]

dΩ = Measure(Triangulation(get_model(mh,1)),qdegree)
x0 = zero(get_fe_space(trials,1))
b = assemble_vector(v -> res(x0,v,dΩ),get_fe_space(tests,1))

coarse_solver = LUSolver()
restrictions, prolongations = setup_transfer_operators(trials,
                                                        qdegree;
                                                        mode=:residual,
                                                        solver=LUSolver());


# GMG as solver 

gmg_solver = GMGLinearSolver(mh,
                      smatrices,
                      prolongations,
                      restrictions,
                      pre_smoothers=smoothers,
                      post_smoothers=smoothers,
                      coarsest_solver=LUSolver(),
                      maxiter=20,
                      rtol=1.0e-8,
                      verbose=true,
                      mode=:preconditioner)
gmg_solver.log.depth += 1
gmg_ns = numerical_setup(symbolic_setup(gmg_solver,A),A)

x = pfill(0.0,partition(axes(A,2)))
r = b - A*x
solve!(x,gmg_ns,r)

# GMG as preconditioner for GMRES

gmg = GMGLinearSolver(mh,
                      smatrices,
                      prolongations,
                      restrictions,
                      pre_smoothers=smoothers,
                      post_smoothers=smoothers,
                      coarsest_solver=LUSolver(),
                      maxiter=3,
                      rtol=1.0e-8,
                      verbose=true,
                      mode=:preconditioner)
gmg.log.depth += 1

gmres_solver = FGMRESSolver(10,gmg;m_add=5,maxiter=30,rtol=1.0e-6,verbose=i_am_main(parts))
gmres_ns = numerical_setup(symbolic_setup(gmres_solver,A),A)

x = pfill(0.0,partition(axes(A,2)))
solve!(x,gmres_ns,b)
