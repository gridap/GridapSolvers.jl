
using Gridap
using GridapSolvers
using LinearAlgebra

using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Geometry
using Gridap.Algebra, Gridap.Arrays, Gridap.Adaptivity
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.LinearSolvers

function l2_norm(uh,dΩ)
  sqrt(sum(∫(uh⋅uh)dΩ))
end

function l2_error(uh,u_exact,dΩ)
  eh = uh - u_exact
  return sqrt(sum(∫(eh⋅eh)dΩ))
end

function add_labels_2d(model)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"newman",[5])
  add_tag_from_tags!(labels,"dirichlet",[collect(1:4)...,6,7,8])
end

function add_labels_3d(model)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"newman",[21])
  add_tag_from_tags!(labels,"dirichlet",[collect(1:20)...,22,23,24,25,26])
end

B = VectorValue(0.0,0.0,1.0)
u_exact(x) = VectorValue(x[1],-x[2],0.0)
p_exact(x) = sum(x)
j_exact(x) = VectorValue(-x[2],-x[1],0.0) + VectorValue(1.0,1.0,0.0)
φ_exact(x) = x[1] + x[2]

Dc = 3
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
ncells = (Dc == 2) ? (4,4) : (2,2,2)
cmodel = CartesianDiscreteModel(domain,ncells)
if Dc == 2
  add_labels_2d(cmodel)
else
  add_labels_3d(cmodel)
end
model = refine(cmodel,2)

order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)
reffe_j = ReferenceFE(raviart_thomas,Float64,order-1)
reffe_φ = ReferenceFE(lagrangian,Float64,order-1)

using Badia2024
using SpecialPolynomials
using Polynomials
reffe_j = Badia2024.SpecialRTReffe(ChebyshevT,order,Dc)

VH = TestFESpace(cmodel,reffe_u;dirichlet_tags="dirichlet")
UH = TrialFESpace(VH,u_exact)
DH = TestFESpace(cmodel,reffe_j;dirichlet_tags="dirichlet")
JH = TrialFESpace(DH,j_exact)
XH = MultiFieldFESpace([UH,JH])
YH = MultiFieldFESpace([VH,DH])

Vh = TestFESpace(model,reffe_u;dirichlet_tags="dirichlet")
Uh = TrialFESpace(Vh,u_exact)
Dh = TestFESpace(model,reffe_j;dirichlet_tags="dirichlet")
Jh = TrialFESpace(Dh,j_exact)
Xh = MultiFieldFESpace([Uh,Jh])
Yh = MultiFieldFESpace([Vh,Dh])

qdegree = 2*order
Ωh = Triangulation(model)
dΩh = Measure(Ωh,qdegree)
ΩH = Triangulation(cmodel)
dΩH = Measure(ΩH,qdegree)
dΩHh = Measure(ΩH,Ωh,qdegree)

Γh = BoundaryTriangulation(model,tags="newman")
dΓh = Measure(Γh,qdegree)
n = get_normal_vector(Γh)

ν = 1.e-8
α = 1.0
f(x) = -(-Δ(u_exact)(x) - cross(j_exact(x),B))
g(x) = j_exact(x) - cross(u_exact(x),B)
σ(x) = ∇(u_exact)(x)

crossB(u,v,dΩ) = ∫(cross(u,B)⋅v)dΩ
Π = GridapSolvers.MultilevelTools.LocalProjectionMap(divergence,reffe_p,qdegree)
Πgraddiv(u,v,dΩ) = ∫(α*Π(u)⋅(∇⋅v))*dΩ
graddiv(u,v,dΩ) = ∫(α*(∇⋅u)⋅(∇⋅v))*dΩ
function a((u,j),(v,d),dΩ)
  c = ∫((-ν)*∇(u)⊙∇(v) + j⋅d)dΩ
  c = c - crossB(u,d,dΩ) + crossB(j,v,dΩ)
  if α > 0.0
    c += Πgraddiv(u,v,dΩ) #+ graddiv(j,d,dΩ)
  end
  return c
end
function a_u(u,v,dΩ) 
  c = ∫(ν*∇(u)⊙∇(v))*dΩ
  if α > 0.0
    c += Πgraddiv(u,v,dΩ)
  end
  return c
end
l((v,d),dΩ,dΓ) = ∫(v⋅f + d⋅g)dΩ + ∫(v⋅(σ⋅n))dΓ

ah(x,y) = a(x,y,dΩh)
aH(x,y) = a(x,y,dΩH)
lh(y) = l(y,dΩh,dΓh)

op_h = AffineFEOperator(ah,lh,Xh,Yh)
Ah = get_matrix(op_h) 
bh = get_vector(op_h)
xh_exact = Ah\bh

AH = assemble_matrix(aH,XH,YH)
Mhh = assemble_matrix((u,v)->∫(u⋅v)*dΩh,Xh,Xh)

uh_exact, jh_exact = FEFunction(Xh,xh_exact)
l2_error(uh_exact,u_exact,dΩh)
l2_error(jh_exact,j_exact,dΩh)
l2_norm(∇⋅uh_exact,dΩh)
l2_norm(∇⋅jh_exact,dΩh)

PD = PatchDecomposition(model)
smoother = RichardsonSmoother(BlockJacobiSolver(Xh,PD),10,ν)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

function project_f2c(rh)
  Qrh = Mhh\rh
  uh, jh = FEFunction(Yh,Qrh)
  ll((v,d)) = ∫(v⋅uh + d⋅jh)*dΩHh
  assemble_vector(ll,YH)
end
function interp_c2f(xH)
  get_free_dof_values(interpolate(FEFunction(YH,xH),Yh))
end

############################################################################################

patches_mask = PatchBasedSmoothers.get_coarse_node_mask(model,model.glue)
Ih = PatchFESpace(Vh,PD,reffe_u;conformity=H1Conformity(),patches_mask=patches_mask)

Ωp = Triangulation(PD)
dΩp = Measure(Ωp,qdegree)
ap(x,y) = a_u(x,y,dΩp)
Ai = assemble_matrix(ap,Ih,Ih)

function P1(dxH)
  interp_c2f(dxH)
end
function R1(rh)
  project_f2c(rh)
end

function P2(dxH)
  xh = interpolate(FEFunction(YH,dxH),Yh)
  dxh = get_free_dof_values(xh)

  uh, jh = xh
  r̃h = assemble_vector(v -> Πgraddiv(uh,v,dΩp),Ih)
  dx̃ = Ai\r̃h
  Pdxh = zero_free_values(Xh)
  Pdxh_u = Gridap.MultiField.restrict_to_field(Xh,Pdxh,1)
  PatchBasedSmoothers.inject!(Pdxh_u,Ih,dx̃)
  y = dxh - Pdxh
  return y
end
function R2(rh)
  rh_u = Gridap.MultiField.restrict_to_field(YH,rh,1)
  r̃h = zero_free_values(Ih)
  PatchBasedSmoothers.prolongate!(r̃h,Ih,rh_u)
  dr̃h = Ai\r̃h

  Pdxh = zero_free_values(Vh)
  PatchBasedSmoothers.inject!(Pdxh,Ih,dr̃h)
  uh = FEFunction(Vh,Pdxh)
  ll((v,q)) = Πgraddiv(uh,v,dΩh)
  drh = assemble_vector(ll,Yh)
  rH = project_f2c(rh - drh)
  return rH
end

############################################################################################

xh = zeros(size(Ah,2))
rh = bh - Ah*xh
niters = 10

iter = 0
error0 = norm(rh)
error = error0
e_rel = error/error0
while iter < niters && e_rel > 1.0e-10
  println("Iter $iter:")
  println(" > Initial: ", norm(rh))

  solve!(xh,smoother_ns,rh)
  println(" > Pre-smoother: ", norm(rh))

  rH = R2(rh)
  qH = AH\rH
  qh = P2(qH)

  rh = rh - Ah*qh
  xh = xh + qh
  println(" > Post-correction: ", norm(rh))

  solve!(xh,smoother_ns,rh)

  iter += 1
  error = norm(rh)
  e_rel = error/error0
  println(" > Final: ",error, " - ", e_rel)
end

uh, jh = FEFunction(Xh,xh)
l2_error(uh,u_exact,dΩh)
l2_error(jh,j_exact,dΩh)
