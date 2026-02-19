using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function get_patch_smoothers(tests,biform;w=0.2,niter=5)
  nlevs = num_levels(tests)
  smoothers = map(view(tests,1:nlevs-1)) do test
    Vh = get_fe_space(test)
    model = get_background_model(get_triangulation(Vh))
    ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
    Ωp  = Geometry.PatchTriangulation(model,ptopo)
    # Λp = Geometry.PatchSkeletonTriangulation(model,ptopo)
    # Γp  = Geometry.PatchBoundaryTriangulation(model,ptopo)

    # Ω = Triangulation(model)
    # Λp  = Geometry.PatchSkeletonTriangulation(Ω,ptopo)
    # Γp  = Geometry.PatchBoundaryTriangulation(Ω,ptopo)

    Λp = Skeleton(Ωp)
    Γp = Boundary(Ωp)
    ap = (u,v) -> biform(u,v,Ωp,Λp,Γp)
    solver = PatchBasedSmoothers.PatchSolver(ptopo,Vh,Vh,ap;assembly=:star,collect_factorizations=true)
    if w > 0.0
      return RichardsonSmoother(solver,niter,w)
    else
      return FGMRESSolver(niter,solver;maxiter=niter)
    end
  end
  return smoothers
end

function get_bilinear_form(mh_lev,biform)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  Λ = Skeleton(model)
  Γ = Boundary(model)
  return (u,v) -> biform(u,v,Ω,Λ,Γ)
end

parts = with_debug() do distribute
  distribute(LinearIndices((1,)))
end

Dc = 2
nx = 8
nc = Tuple(fill(nx,Dc))
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)

cmodel = CartesianDiscreteModel(domain,nc)
model = Gridap.Adaptivity.refine(cmodel)
mh = ModelHierarchy([model,cmodel])

Ω = Triangulation(model)
dΩ = Measure(Ω,5)

# FE spaces
order = 1
qdegree = 2*(order+1)
reffe_u = ReferenceFE(raviart_thomas,Float64,order-1)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:Q)

function u_exact(x)
  # if Dc == 2
  #   return VectorValue(x[1],-x[2])
  # else
  #   return VectorValue(2*x[1],-x[2],-x[3])
  # end
  o = ifelse(Dc==2,Point(1.,0.),Point(1.,0.,0.))
  z = ifelse(Dc==2,Point(0.,0.),Point(0.,0.,0.))
  return ifelse(x[Dc] > 0.99,o,z)
end
_p_exact(x) = sum(x)
p_mean = sum(∫(_p_exact)dΩ)
p_exact(x) = sum(x) - p_mean

u_cf = CellField(u_exact,Ω)
Divu = CellField(x -> divergence(u_exact)(x), Ω)
sum(∫(Divu ⋅ Divu)dΩ)

tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags="boundary")
trials_u = TrialFESpace(tests_u,u_exact)
U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

# Weak formulation
α = 1.e2
μ = order*(order+1)
h = 1/nx
f(x) = zero(VectorValue{Dc,Float64})#-Δ(u_exact)(x) + ∇(p_exact)(x)
function biform_u(u,v,Ω,Λ,Γ)
  dΩ = Measure(Ω,qdegree)
  dΛ = Measure(Λ,qdegree)
  dΓ = Measure(Γ,qdegree)
  n_Λ = get_normal_vector(Λ)
  n_Γ = get_normal_vector(Γ)
  h_Λ = CellField(get_array(∫(1)dΛ),Λ)
  h_Γ = CellField(get_array(∫(1)dΓ),Γ)

  c = ∫(∇(v)⊙∇(u) + α*(∇⋅v)*(∇⋅u))*dΩ +
      ∫((μ/h_Γ)*(v⋅u) - v⋅(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))⋅u )*dΓ +
      ∫(
        (μ/h_Λ)*(jump(v⊗n_Λ)⊙jump(u⊗n_Λ)) -
          jump(v⊗n_Λ)⊙mean(∇(u)) -
          mean(∇(v))⊙jump(u⊗n_Λ)
      )*dΛ
  return c
end
function biform((u,p),(v,q),Ω,Λ,Γ)
  dΩ = Measure(Ω,qdegree)
  return biform_u(u,v,Ω,Λ,Γ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
end
function liform_u(v,Ω,Λ,Γ)
  dΩ = Measure(Ω,qdegree)
  dΓ = Measure(Γ,qdegree)
  n_Γ = get_normal_vector(Γ)
  h_Γ = CellField(get_array(∫(1)dΓ),Γ)
  return ∫(v⋅f)dΩ + ∫( (μ/h_Γ)*(v⋅u_exact) - (n_Γ⋅∇(v))⋅u_exact)*dΓ
end
liform((v,q),Ω,Λ,Γ) = liform_u(v,Ω,Λ,Γ)

Ω = Triangulation(model)
Λ = Skeleton(model)
Γ = Boundary(model)

a(u,v) = biform(u,v,Ω,Λ,Γ)
l(v) = liform(v,Ω,Λ,Γ)
op = AffineFEOperator(a,l,X,Y)
A, b = get_matrix(op), get_vector(op);

uh, ph = solve(op)

dΩ = Measure(Ω,qdegree)
eu = uh - u_exact
ep = ph - p_exact
err_u = sum(∫(eu⋅eu)dΩ)
err_p = sum(∫(ep*ep)dΩ)

writevtk(Ω,"test/_dev/sol"; cellfields=["uh" => uh, "eu" => eu, "ph" => ph, "ep" => ep], append=false)

# GMG Solver for u
biforms = map(mhl -> get_bilinear_form(mhl,biform_u),mh)

smoothers = get_patch_smoothers(
  trials_u,biform_u; w=0.2, niter=10
);
prolongations = setup_prolongation_operators(
  tests_u,qdegree;mode=:residual
);
restrictions = setup_restriction_operators(
  tests_u,qdegree;mode=:residual,solver=CGSolver(JacobiLinearSolver())
);

gmg = GMGLinearSolver(
  trials_u,tests_u,biforms,
  prolongations,restrictions,
  pre_smoothers=smoothers,
  post_smoothers=smoothers,
  coarsest_solver=LUSolver(),
  maxiter=2,mode=:preconditioner,
  verbose=i_am_main(parts),
  cycle_type = :f_cycle,
);
gmg.log.depth = 4

# Solver
solver_u = FGMRESSolver(10,gmg;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts));
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
