using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces, Gridap.Polynomials
using Gridap.CellData, Gridap.MultiField, Gridap.Arrays, Gridap.Fields, Gridap.Helpers

using PartitionedArrays
using GridapDistributed
using GridapPETSc

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: NonlinearSystemBlock, LinearSystemBlock, BiformBlock, TriformBlock
using GridapSolvers.BlockSolvers: BlockTriangularSolver, BlockDiagonalSolver
using GridapSolvers.MultilevelTools: get_level_parts

function HuntModelHierarchy(
  parts,np_per_level,nc
)
  return CartesianModelHierarchy(
    parts,np_per_level,(-1,1,-1,1,0,0.2),nc;
    add_labels! = add_labels_hunt!,
    isperiodic  = (false,false,true),
    #map = hunt_stretch_map(1.0,sol.Ha,1.0,1.0,adapt),
    nrefs = (2,2,2)
  )
end

function add_labels_hunt!(labels)
  add_tag_from_tags!(labels,"noslip",[1:20...,23,24,25,26])
  add_tag_from_tags!(labels,"insulating",[1:20...,25,26])
  add_tag_from_tags!(labels,"topbottom",[21,22])
end

function get_patch_smoothers(
  mh,tests,biform,qdegree;
  w=0.2,
  niter=20,
  is_nonlinear=false,
  patch_decompositions = PatchDecomposition(mh)
)
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),patch_decompositions,patch_spaces) do tests, PD, Ph
    Vh = get_fe_space(tests)
    Ω  = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    ap = is_nonlinear ? (u,du,dv) -> biform(u,du,dv,dΩ) : (u,v) -> biform(u,v,dΩ)
    patch_solver = PatchBasedLinearSolver(ap,Ph,Vh;is_nonlinear=is_nonlinear)
    if w < 0
      solver = GMRESSolver(niter;Pr=patch_solver,maxiter=niter)
      patch_smoother = RichardsonSmoother(solver,1,1.0)
    else
      patch_smoother = RichardsonSmoother(patch_solver,niter,w)
    end
    return patch_smoother
  end
  return smoothers
end

function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

np = (2,2,1)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

mh = HuntModelHierarchy(parts,[np,np],(4,4,3))

order = 2
qdegree = 2*(order+1)
reffe = ReferenceFE(raviart_thomas,Float64,order-1)
tests = TestFESpace(mh,reffe,dirichlet_tags=["insulating"]);

trials = TrialFESpace(tests,VectorValue(0.0,0.0,0.0));
J, D = get_fe_space(trials,1), get_fe_space(tests,1)

η = 100.0
mass(x,v_x,dΩ) = ∫(v_x⋅x)dΩ
graddiv(x,v_x,dΩ) = ∫(η*divergence(v_x)⋅divergence(x))dΩ
biform(j,v_j,dΩ) = mass(j,v_j,dΩ) + graddiv(j,v_j,dΩ)

biforms = map(mhl -> get_bilinear_form(mhl,biform,qdegree),mh)
smoothers = get_patch_smoothers(
  mh,trials,biform,qdegree; w=0.2
)
prolongations = setup_prolongation_operators(
  trials,qdegree;mode=:residual,solver=LUSolver()
)
restrictions = setup_restriction_operators(
  trials,qdegree;mode=:residual,solver=LUSolver()
)
gmg = GMGLinearSolver(
  mh,trials,tests,biforms,
  prolongations,restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter = 3, mode = :preconditioner,
  verbose = i_am_main(parts)
)

solver = FGMRESSolver(5,gmg;maxiter=10,rtol=1.e-8,verbose=i_am_main(parts))

Ω = Triangulation(get_model(mh,1))
dΩ = Measure(Ω,qdegree)

ah(j,d) = biform(j,d,dΩ)
Ah = assemble_matrix(ah,D,J)

b = allocate_in_range(Ah)
fill!(b,1.0)

ns = numerical_setup(symbolic_setup(solver,Ah),Ah)
x = allocate_in_domain(Ah)
fill!(x,0.0)
solve!(x,ns,b)
