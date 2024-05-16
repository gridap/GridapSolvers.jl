using LinearAlgebra

using Gridap
using PartitionedArrays
using GridapDistributed
using GridapP4est

using Gridap.FESpaces

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

u_exact_2d(x) = VectorValue(x[1]^2,-2.0*x[1]*x[2])
u_exact_3d(x) = VectorValue(x[1]^2,-2.0*x[1]*x[2],1.0)

function Gridap.cross(a::VectorValue{2},b::VectorValue{3})
  @assert iszero(b[1]) && iszero(b[2])
  VectorValue(a[2]*b[3],-a[1]*b[3])
end

function get_patch_smoothers(
  mh,tests,biform,qdegree;
  w=0.2,
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
    patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh;is_nonlinear=is_nonlinear)
    return RichardsonSmoother(patch_smoother,5,w)
  end
  return smoothers
end

function get_patch_smoothers_bis(
  mh,tests,biform,qdegree;
  niter = 10,
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
    patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh;is_nonlinear=is_nonlinear)
    gmres = GMRESSolver(niter;Pr=patch_smoother,maxiter=niter,atol=1e-14,rtol=1.e-10,verbose=false);
    return RichardsonSmoother(gmres,1,1.0)
  end
  return smoothers
end

function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

############################################################################################

Dc = 3
np = 1
nc = Tuple(fill(4,Dc))
np_per_level = [np,np]
parts = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end

domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
mh = CartesianModelHierarchy(parts,np_per_level,domain,nc)

B = VectorValue(0.0,0.0,1.0)
u_exact(x) = (Dc == 2) ? u_exact_2d(x) : u_exact_3d(x)
j_exact(x) = cross(u_exact(x),B)
f(x) = -Δ(u_exact)(x) - cross(j_exact(x),B)

order = 2
qdegree = 2*(order+1)
reffe_h1 = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_hdiv = ReferenceFE(raviart_thomas,Float64,order-1)

tests_u = TestFESpace(mh,reffe_h1,dirichlet_tags="boundary");
tests_j = TestFESpace(mh,reffe_hdiv,dirichlet_tags="boundary");
trials_u = TrialFESpace(tests_u,u_exact);
trials_j = TrialFESpace(tests_j,j_exact);
tests = MultiFieldFESpace([tests_u,tests_j]);
trials = MultiFieldFESpace([trials_u,trials_j]);

Ha = 1.0e3
β = 1/Ha^2  # Laplacian coefficient
η = 1000

poly = (Dc == 2) ? QUAD : HEX
Π = LocalProjectionMap(poly,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
mass(x,v_x,dΩ) = ∫(v_x⋅x)dΩ
lap(x,v_x,dΩ) = ∫(β*∇(v_x)⊙∇(x))dΩ
graddiv(x,v_x,dΩ) = ∫(η*divergence(v_x)⋅divergence(x))dΩ
Qgraddiv(x,v_x,dΩ) = ∫(η*Π(divergence(v_x))⋅Π(divergence(x)))dΩ
crossB(x,v_x,dΩ) = ∫(v_x⋅cross(x,B))dΩ

biform_u(u,v_u,dΩ) = lap(u,v_u,dΩ) + Qgraddiv(u,v_u,dΩ)
biform_j(j,v_j,dΩ) = mass(j,v_j,dΩ) + graddiv(j,v_j,dΩ)
biform((u,j),(v_u,v_j),dΩ) = biform_u(u,v_u,dΩ) + biform_j(j,v_j,dΩ) - crossB(u,v_j,dΩ) - crossB(j,v_u,dΩ)
liform((v_u,v_j),dΩ) = ∫(v_u⋅f)dΩ

rhs((u,j),(v_u,v_j),dΩ) = Qgraddiv(u,v_u,dΩ) + graddiv(j,v_j,dΩ)
rhs_u(u,v_u,dΩ) = Qgraddiv(u,v_u,dΩ)

smatrices, A, b = compute_hierarchy_matrices(trials,tests,biform,liform,qdegree);
smoothers = get_patch_smoothers_bis(mh,tests,biform,qdegree);
prolongations = setup_patch_prolongation_operators(tests,biform,biform,qdegree);
restrictions = setup_patch_restriction_operators(tests,prolongations,biform,qdegree);

gmg = GMGLinearSolver(
  mh,smatrices,prolongations,restrictions,
  pre_smoothers=smoothers,post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter=4,rtol=1.0e-8,
  verbose=i_am_main(parts),mode=:preconditioner
);
gmg.log.depth += 1

# Standalone GMG
gmg_ns = numerical_setup(symbolic_setup(gmg,A),A)
x = pfill(0.0,partition(axes(A,2)))
r = b - A*x
solve!(x,gmg_ns,r)

# FGMRES + GMG
#solver = FGMRESSolver(10,gmg;m_add=5,maxiter=30,atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts));
#ns = numerical_setup(symbolic_setup(solver,A),A);

#x = pfill(0.0,partition(axes(A,2)));
#solve!(x,ns,b)
