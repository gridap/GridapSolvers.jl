
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

VH = TestFESpace(cmodel,reffe_u;dirichlet_tags="dirichlet")
UH = TrialFESpace(VH,u_exact)
QH = TestFESpace(cmodel,reffe_p,conformity=:L2)
DH = TestFESpace(cmodel,reffe_j;dirichlet_tags="dirichlet")
JH = TrialFESpace(DH,j_exact)
ΦH = TestFESpace(cmodel,reffe_φ,conformity=:L2)

XH = MultiFieldFESpace([UH,QH,JH,ΦH])
YH = MultiFieldFESpace([VH,QH,DH,ΦH])

Vh = TestFESpace(model,reffe_u;dirichlet_tags="dirichlet")
Uh = TrialFESpace(Vh,u_exact)
Qh = TestFESpace(model,reffe_p,conformity=:L2)
Dh = TestFESpace(model,reffe_j;dirichlet_tags="dirichlet")
Jh = TrialFESpace(Dh,j_exact)
Φh = TestFESpace(model,reffe_φ,conformity=:L2)
Xh = MultiFieldFESpace([Uh,Qh,Jh,Φh])
Yh = MultiFieldFESpace([Vh,Qh,Dh,Φh])

qdegree = 2*order
Ωh = Triangulation(model)
dΩh = Measure(Ωh,qdegree)
ΩH = Triangulation(cmodel)
dΩH = Measure(ΩH,qdegree)
dΩHh = Measure(ΩH,Ωh,qdegree)

Γh = BoundaryTriangulation(model,tags="newman")
dΓh = Measure(Γh,qdegree)
n = get_normal_vector(Γh)

α = -1.0
I_tensor = one(TensorValue{Dc,Dc,Float64})
f(x) = -Δ(u_exact)(x) - ∇(p_exact)(x) #- cross(j_exact(x),B)
g(x) = j_exact(x) - ∇(φ_exact)(x) #- cross(u_exact(x),B)
σ(x) = ∇(u_exact)(x) + p_exact(x)*I_tensor
γ(x) = φ_exact(x)*I_tensor
crossB(u,v,dΩ) = ∫(cross(u,B)⋅v)dΩ
graddiv(u,v,dΩ) = ∫((∇⋅u)⋅(∇⋅v))*dΩ
function a((u,p,j,φ),(v,q,d,ψ),dΩ)
  c = ∫(∇(u)⊙∇(v) + (∇⋅v)*p - (∇⋅u)*q)dΩ
  c = c + ∫(j⋅d + (∇⋅d)*φ - (∇⋅j)*ψ)dΩ
#  c = c - crossB(u,d,dΩ) - crossB(j,v,dΩ)
  if α > 0.0
    c += graddiv(u,v,dΩ) + graddiv(j,d,dΩ)
  end
  return c
end
l((v,q,d,ψ),dΩ,dΓ) = ∫(v⋅f + d⋅g)dΩ + ∫(v⋅(σ⋅n) + d⋅(γ⋅n))dΓ

ah(x,y) = a(x,y,dΩh)
aH(x,y) = a(x,y,dΩH)
lh(y) = l(y,dΩh,dΓh)

op_h = AffineFEOperator(ah,lh,Xh,Yh)
Ah = get_matrix(op_h) 
bh = get_vector(op_h)
xh_exact = Ah\bh

AH = assemble_matrix(aH,XH,YH)
Mhh = assemble_matrix((u,v)->∫(u⋅v)*dΩh,Xh,Xh)

uh_exact, ph_exact, jh_exact, φh_exact = FEFunction(Xh,xh_exact)
l2_error(uh_exact,u_exact,dΩh)
l2_error(ph_exact,p_exact,dΩh)
l2_error(jh_exact,j_exact,dΩh)
l2_error(φh_exact,φ_exact,dΩh)
l2_norm(∇⋅uh_exact,dΩh)
l2_norm(∇⋅jh_exact,dΩh)

PD = PatchDecomposition(model)
smoother = RichardsonSmoother(BlockJacobiSolver(Xh,PD),10,0.05)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

function project_f2c(rh)
  Qrh = Mhh\rh
  uh, ph, jh, φh = FEFunction(Yh,Qrh)
  ll((v,q,d,ψ)) = ∫(v⋅uh + q*ph + d⋅jh + ψ⋅φh)*dΩHh
  assemble_vector(ll,YH)
end
function interp_c2f(xH)
  get_free_dof_values(interpolate(FEFunction(YH,xH),Yh))
end

xh = randn(size(Ah,2))
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

  rH = project_f2c(rh)
  qH = AH\rH
  qh = interp_c2f(qH)

  rh = rh - Ah*qh
  xh = xh + qh
  println(" > Post-correction: ", norm(rh))

  solve!(xh,smoother_ns,rh)

  iter += 1
  error = norm(rh)
  e_rel = error/error0
  println(" > Final: ",error, " - ", e_rel)
end

uh, ph, jh, φh = FEFunction(Xh,xh)
l2_error(uh,u_exact,dΩh)
l2_error(ph,p_exact,dΩh)
l2_error(jh,j_exact,dΩh)
l2_error(φh,φ_exact,dΩh)
