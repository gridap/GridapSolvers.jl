using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.Arrays

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

u_h1(x) = x[1] + x[2]
# u_h1(x) = x[1]*(1-x[1])*x[2]*(1-x[2])
#u_hdiv(x) = VectorValue(x[1],x[2])
u_hdiv(x) = VectorValue([x[1]*(1.0-x[1]),-x[2]*(1.0-x[2])])

a_h1(u,v,dΩ) = ∫(u⋅v)*dΩ
a_hdiv(u,v,dΩ) = ∫(u⋅v + divergence(u)⋅divergence(v))*dΩ

conf  = :hdiv
order = 1
poly  = QUAD

cmodel = CartesianDiscreteModel((0,1,0,1),(16,16))
if poly == TRI
  cmodel = simplexify(cmodel)
end
fmodel = refine(cmodel)

Ωh = Triangulation(fmodel)
ΩH = Triangulation(cmodel)

qdegree = 2*(order+1)
dΩh = Measure(Ωh,qdegree)
dΩH = Measure(ΩH,qdegree)
dΩHh = Measure(ΩH,Ωh,qdegree)

if conf == :h1
  u_bc  = u_h1
  conformity = H1Conformity()
  reffe = ReferenceFE(lagrangian,Float64,order)
  ah(u,v) = a_h1(u,v,dΩh)
  aH(u,v) = a_h1(u,v,dΩH)
  f = zero(Float64)
else
  u_bc  = u_hdiv
  conformity = DivConformity()
  reffe = ReferenceFE(raviart_thomas,Float64,order)
  ah(u,v) = a_hdiv(u,v,dΩh)
  aH(u,v) = a_hdiv(u,v,dΩH)
  f = zero(VectorValue{2,Float64})
end
lh(v) = ∫(v⋅f)*dΩh
lH(v) = ∫(v⋅f)*dΩH

VH = TestFESpace(cmodel,reffe,dirichlet_tags="boundary")
UH = TrialFESpace(VH,u_bc)
Vh = TestFESpace(fmodel,reffe,dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u_bc)

oph = AffineFEOperator(ah,lh,Uh,Vh)
opH = AffineFEOperator(aH,lH,UH,VH)

xh_star = get_free_dof_values(solve(oph))
xH_star = get_free_dof_values(solve(opH))

Ah, bh = get_matrix(oph), get_vector(oph);
AH, bH = get_matrix(opH), get_vector(opH);


Mhh = assemble_matrix((u,v)->∫(u⋅v)*dΩh,Vh,Vh)

function compute_MhH()
  MhH = zeros(num_free_dofs(Vh),num_free_dofs(VH))
  xHi = fill(0.0,num_free_dofs(VH))
  for iH in 1:num_free_dofs(VH)
    fill!(xHi,0.0); xHi[iH] = 1.0
    vHi = FEFunction(VH,xHi)
    vH = assemble_vector((v)->∫(v⋅vHi)*dΩh,Vh)
    MhH[:,iH] .= vH
  end
  return MhH
end

MhH = compute_MhH()


# Projection operators 

function Λ_project(xh)
  uh = FEFunction(Vh,xh)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩh,v->∫(v⋅uh + divergence(v)⋅divergence(uh))*dΩh,Vh,Vh)
  return get_matrix(op)\get_vector(op)
end

function project_c2f(xH)
  uH = FEFunction(VH,xH)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩh,v->∫(v⋅uH)*dΩh,Vh,Vh)
  return get_matrix(op)\get_vector(op)
end

function Λ_project_c2f(xH)
  uH = FEFunction(VH,xH)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩh,v->∫(v⋅uH + divergence(v)⋅divergence(uH))*dΩh,Vh,Vh)
  return get_matrix(op)\get_vector(op)
end

function Λ_project_f2c(xh)
  uh = FEFunction(Vh,xh)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩH,v->∫(v⋅uh + divergence(v)⋅divergence(uh))*dΩHh,VH,VH)
  return get_matrix(op)\get_vector(op)
end

function project_f2c(xh)
  uh = FEFunction(Vh,xh)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩH,v->∫(v⋅uh)*dΩHh,VH,VH)
  return get_matrix(op)\get_vector(op)
end

function interp_f2c(xh)
  get_free_dof_values(interpolate(FEFunction(Vh,xh),VH))
end

function dotH(a::AbstractVector,b::AbstractVector)
  _a = FEFunction(VH,a)
  _b = FEFunction(VH,b)
  dotH(_a,_b)
end

function dotH(a,b)
  sum(∫(a⋅b)*dΩH)
end

function doth(a::AbstractVector,b::AbstractVector)
  _a = FEFunction(Vh,a)
  _b = FEFunction(Vh,b)
  doth(_a,_b)
end

function doth(a,b)
  sum(∫(a⋅b)*dΩh)
end

# Patch Decomposition

PD = PatchDecomposition(fmodel)
Ph = PatchFESpace(Vh,PD,reffe;conformity)
Ωp = Triangulation(PD)
dΩp = Measure(Ωp,qdegree)

if conf == :h1
  smoother = RichardsonSmoother(JacobiLinearSolver(),5,0.6)
else
  ap(u,v) = a_hdiv(u,v,dΩp)
  smoother = RichardsonSmoother(PatchBasedLinearSolver(ap,Ph,Vh),10,0.2)
  Ap = assemble_matrix(ap,Ph,Ph)
end
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

function smooth!(x,r)
  A  = smoother_ns.A
  Ap = smoother_ns.Mns.Ap_ns.A

  dx  = smoother_ns.dx
  rp  = smoother_ns.Mns.caches[1]
  dxp = smoother_ns.Mns.caches[2]
  w, w_sums = smoother_ns.Mns.weights
  w_sums = fill(1.0,length(w_sums))

  β = 0.4
  niter = 10
  for i in 1:niter
    _r = bh - Λ_project(x)
    PatchBasedSmoothers.prolongate!(rp,Ph,r,w,w_sums)
    dxp = Ap\rp
    PatchBasedSmoothers.inject!(dx,Ph,dxp,w,w_sums)

    x .+= β*dx
    r .-= β*A*dx
  end
end

xh = fill(1.0,size(Ah,2))
rh = bh - Ah*xh
niters = 10

wH = randn(size(AH,2))
wh = project_c2f(wH)

function project_f2c_bis(rh)
  Qrh = Mhh\rh
  uh  = FEFunction(Vh,Qrh)
  assemble_vector(v->∫(v⋅uh)*dΩHh,VH)
end

iter = 0
error = norm(bh - Ah*xh)
while iter < niters && error > 1.0e-8
  println("Iter $iter:")
  println(" > Pre-smoother: ")
  println("    > norm(xh) = ",norm(xh))
  println("    > norm(rh) = ",norm(rh))

  solve!(xh,smoother_ns,rh)

  println(" > Post-smoother: ")
  println("    > norm(xh) = ",norm(xh))
  println("    > norm(rh) = ",norm(rh))

  rH = project_f2c_bis(rh)

  qH = AH\rH
  qh = project_c2f(qH)

  println(" > GMG approximation properties:")
  println("    > (AH*qH,wH)     = ",dotH(AH*qH,wH))
  println("    > (rH,wH)        = ",dotH(rH,wH))
  println("    > (rh,wh)        = ",doth(rh,wh))
  println("    > (Ah*(x-xh),wh) = ",doth(Ah*(xh_star-xh),wh))
  rh = rh - Ah*qh
  xh = xh + qh

  solve!(xh,smoother_ns,rh)

  iter += 1
  error = norm(bh - Ah*xh)
  println(" > error          = ",error)
end

######################################################################################
using GridapSolvers.PatchBasedSmoothers: prolongate!, inject!, compute_weight_operators

xh = fill(1.0,size(Ah,2))
rh = bh - Ah*xh

w, w_sums = compute_weight_operators(Ph,Ph.Vh)

rp = fill(0.0,size(Ap,2))
prolongate!(rp,Ph,rh)

xp = Ap\rp
dxh = fill(0.0,size(Ah,2))
inject!(dxh,Ph,xp,w,w_sums)

_rh = fill(0.0,size(Ah,2))
inject!(_rh,Ph,rp,w,w_sums)

patch_cells = PD.patch_cells
ids_Ph = get_cell_dof_ids(Ph)
ids_Vh = get_cell_dof_ids(Vh)

patch_ids_Ph = Table(ids_Ph,patch_cells.ptrs)
patch_ids_Vh = Table(lazy_map(Reindex(ids_Vh),patch_cells.data),patch_cells.ptrs)


######################################################################################
using LinearAlgebra
function LinearAlgebra.ldiv!(x,ns,b)
  solve!(x,ns,b)
end

using IterativeSolvers
x = zeros(size(Ah,2))
cg!(x,Ah,bh;Pl=smoother_ns.Mns,verbose=true)



######################################################################################

_s  = IS_ConjugateGradientSolver(maxiter=50,reltol=1.e-16,verbose=true)
_ns = numerical_setup(symbolic_setup(_s,Mhh),Mhh)

x = zeros(size(Mhh,2))
_rh = copy(rh)
solve!(x,_ns,_rh)

norm(rh - Mhh*x)
