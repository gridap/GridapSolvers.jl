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

u_exact(x) = VectorValue(-x[1],x[2])
p_exact(x) = x[1] - 0.5

Dc = 2
cmodel = CartesianDiscreteModel((0,1,0,1),(4,4))
labels = get_face_labeling(cmodel)
add_tag_from_tags!(labels,"newman",[5])
add_tag_from_tags!(labels,"dirichlet",[collect(1:4)...,6,7,8])
model = refine(cmodel,2)

order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

VH = TestFESpace(cmodel,reffe_u;dirichlet_tags="dirichlet")
UH = TrialFESpace(VH,u_exact)
QH = TestFESpace(cmodel,reffe_p,conformity=:L2)
XH = MultiFieldFESpace([UH,QH])
YH = MultiFieldFESpace([VH,QH])

Vh = TestFESpace(model,reffe_u;dirichlet_tags="dirichlet")
Uh = TrialFESpace(Vh,u_exact)
Qh = TestFESpace(model,reffe_p,conformity=:L2)
Xh = MultiFieldFESpace([Uh,Qh])
Yh = MultiFieldFESpace([Vh,Qh])

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
α = 10.0
I_tensor = one(TensorValue{Dc,Dc,Float64})
f(x) = -ν*Δ(u_exact)(x) - ∇(p_exact)(x)
σ(x) = ν*∇(u_exact)(x) + p_exact(x)*I_tensor

Π = GridapSolvers.MultilevelTools.LocalProjectionMap(divergence,reffe_p,qdegree)
#graddiv(u,v,dΩ) = ∫(α*(∇⋅u)⋅(∇⋅v))*dΩ
graddiv(u,v,dΩ) = ∫(α*Π(u)⋅(∇⋅v))*dΩ
function a_u(u,v,dΩ) 
  c = ∫(ν*∇(u)⊙∇(v))*dΩ
  if α > 0.0
    c += graddiv(u,v,dΩ)
  end
  return c
end
function a((u,p),(v,q),dΩ)
  c = ∫(ν*∇(u)⊙∇(v) + (∇⋅v)*p - (∇⋅u)*q)dΩ
  if α > 0.0
    c += graddiv(u,v,dΩ)
  end
  return c
end
l((v,q),dΩ,dΓ) = ∫(v⋅f)dΩ + ∫(v⋅(σ⋅n) )dΓ

ah(x,y) = a(x,y,dΩh)
aH(x,y) = a(x,y,dΩH)
lh(y) = l(y,dΩh,dΓh)

op_h = AffineFEOperator(ah,lh,Xh,Yh)
Ah = get_matrix(op_h) 
bh = get_vector(op_h)
xh_exact = Ah\bh

AH = assemble_matrix(aH,XH,YH)
Mhh = assemble_matrix((u,v)->∫(u⋅v)*dΩh,Xh,Xh)

uh_exact, ph_exact = FEFunction(Xh,xh_exact)
l2_error(uh_exact,u_exact,dΩh)
l2_error(ph_exact,p_exact,dΩh)
l2_norm(∇⋅uh_exact,dΩh)

PD = PatchDecomposition(model)
smoother = RichardsonSmoother(VankaSolver(Xh,PD),10,0.1)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

function project_f2c(rh)
  Qrh = Mhh\rh
  uh, ph = FEFunction(Yh,Qrh)
  ll((v,q)) = ∫(v⋅uh + q*ph)*dΩHh
  assemble_vector(ll,YH)
end
function interp_c2f(xH)
  get_free_dof_values(interpolate(FEFunction(YH,xH),Yh))
end

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

  uh, ph = xh
  r̃h = assemble_vector(v -> graddiv(uh,v,dΩp),Ih)
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
  ll((v,q)) = graddiv(uh,v,dΩh)
  drh = assemble_vector(ll,Yh)
  rH = project_f2c(rh - drh)
  return rH
end


xh = randn(size(Ah,2))
rh = bh - Ah*xh
niters = 20

iter = 0
error0 = norm(rh)
error = error0
e_rel = error/error0
while iter < niters && e_rel > 1.0e-8
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

uh, ph = FEFunction(Xh,xh)
l2_error(uh,u_exact,dΩh)
l2_error(ph,p_exact,dΩh)
l2_norm(∇⋅uh,dΩh)
