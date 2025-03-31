using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function get_patch_smoothers(mh,tests,biform,patch_decompositions,qdegree)
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),patch_decompositions,patch_spaces) do tests, PD, Ph
    Vh = get_fe_space(tests)
    Ω  = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    ap = (u,v) -> biform(u,v,dΩ)
    patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh)
    return RichardsonSmoother(patch_smoother,10,0.2)
  end
  return smoothers
end

function get_bilinear_form(mh_lev,biform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,v) -> biform(u,v,dΩ)
end

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"walls",[1,5,7,2,8])
  add_tag_from_tags!(labels,"right",[2,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
  add_tag_from_tags!(labels,"walls",[1,2,9,13,14,17,18,21,23,25,26,3,4,10,19,20,24])
  add_tag_from_tags!(labels,"right",[3,4,10,19,20,24])
end

np = (1,1)
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

# Geometry
Dc = 2
nc = Tuple(fill(8,Dc))
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
mh = CartesianModelHierarchy(parts,[np,np],domain,nc;add_labels! = add_labels!)
model = get_model(mh,1)

# FE spaces
order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

u_wall = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["walls","top"]);
trials_u = TrialFESpace(tests_u,[u_wall,u_top]);
U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

# Weak formulation
α = 1.e2
f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
poly = (Dc==2) ? QUAD : HEX
Π_Qh = LocalProjectionMap(divergence,lagrangian,Float64,order-1;space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(u,dΩ)⋅Π_Qh(v,dΩ))dΩ
biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ + graddiv(u,v,dΩ)
biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
liform((v,q),dΩ) = ∫(v⋅f)dΩ

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,X,Y)
A, b = get_matrix(op), get_vector(op);

# GMG Solver for u
biforms = map(mhl -> get_bilinear_form(mhl,biform_u,qdegree),mh)
patch_decompositions = PatchDecomposition(mh)
smoothers = get_patch_smoothers(
  mh,tests_u,biform_u,patch_decompositions,qdegree
);
prolongations = setup_patch_prolongation_operators(
  tests_u,biform_u,graddiv,qdegree
);
restrictions = setup_patch_restriction_operators(
  tests_u,prolongations,graddiv,qdegree;solver=LUSolver()
);

gmg = GMGLinearSolver(
  trials_u,tests_u,biforms,
  prolongations,restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter=4,mode=:preconditioner,verbose=i_am_main(parts)
);

# Solver
solver_u = gmg;
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts));
solver_u.log.depth = 2
solver_p.log.depth = 2

diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
bblocks = map(CartesianIndices((2,2))) do I
  (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
end
coeffs = [1.0 1.0;
          0.0 1.0]  
P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts))
ns = numerical_setup(symbolic_setup(solver,A),A)

x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)
xh = FEFunction(X,x);

r = allocate_in_range(A)
mul!(r,A,x)
r .-= b
norm(r) < 1.e-6
