using Gridap
using GridapSolvers
using LinearAlgebra
using FillArrays

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

function mock_model()
  coords = [VectorValue(0.0,0.0),VectorValue(1.0,0.0),VectorValue(0.5,sqrt(3.0)/2.0)]
  polys = [TRI]
  cell_types = Int8[1]
  conn = Table([[1,2,3]])
  reffes = map(x->LagrangianRefFE(Float64,x,1),polys)

  ref_grid = UnstructuredGrid(coords,conn,reffes,cell_types;has_affine_map=true)
  model = UnstructuredDiscreteModel(ref_grid)

  labels = get_face_labeling(model)
  labels.d_to_dface_to_entity[1] .= [1,2,3]
  labels.d_to_dface_to_entity[2] .= [4,5,6]
  labels.d_to_dface_to_entity[3] .= [7]
  for i in 1:7
    push!(labels.tag_to_entities,Int32[i])
    push!(labels.tag_to_name,"tag_$(i)")
  end
  push!(labels.tag_to_entities,Int32[1,2,3,4,5,6])
  push!(labels.tag_to_name,"boundary")
  push!(labels.tag_to_entities,Int32[7])
  push!(labels.tag_to_name,"interior")

  return model
end

u_exact(x) = VectorValue(-x[1],x[2])

Dc = 2
#cmodel = simplexify(CartesianDiscreteModel((0,1,0,1),(2,2)))
cmodel = refine(mock_model()).model

labels = get_face_labeling(cmodel)
#add_tag_from_tags!(labels,"newman",[5])
#add_tag_from_tags!(labels,"dirichlet",[collect(1:4)...,6,7,8])
add_tag_from_tags!(labels,"dirichlet",[collect(1:3)...,4,5])
add_tag_from_tags!(labels,"newman",[6])
model = refine(cmodel)

order = 2
rrule = Adaptivity.BarycentricRefinementRule(TRI)
reffes = Fill(LagrangianRefFE(VectorValue{2,Float64},TRI,order),Adaptivity.num_subcells(rrule))
reffe_u = Adaptivity.MacroReferenceFE(rrule,reffes)

VH = TestFESpace(cmodel,reffe_u;dirichlet_tags="dirichlet")
UH = TrialFESpace(VH,u_exact)

Vh = TestFESpace(model,reffe_u;dirichlet_tags="dirichlet")
Uh = TrialFESpace(Vh,u_exact)

qdegree = 2*order
quad = Quadrature(rrule,qdegree)
Ωh = Triangulation(model)
dΩh = Measure(Ωh,quad)
ΩH = Triangulation(cmodel)
dΩH = Measure(ΩH,quad)
dΩHh = Measure(ΩH,Ωh,quad)

Γh = Boundary(model.model)
dΓh = Measure(Γh,4*order)
nh = get_normal_vector(Γh)

ν = 1.e-2
α = 10.0
f(x) = -ν*Δ(u_exact)(x)
graddiv(u,v,dΩ) = ∫(α*(∇⋅u)⋅(∇⋅v))*dΩ
function a(u,v,dΩ)
  c = ∫(ν*∇(u)⊙∇(v))*dΩ
  if α > 0.0
    c += graddiv(u,v,dΩ)
  end
  return c
end
l(v,dΩ) = ∫(v⋅f)dΩ

∇u_exact(x) = ∇(u_exact)(x) 
ah(x,y) = a(x,y,dΩh)
aH(x,y) = a(x,y,dΩH)
lh(v) = l(v,dΩh) + ∫(ν*v⋅(∇u_exact⋅nh))dΓh

op_h = AffineFEOperator(ah,lh,Uh,Vh)
Ah = get_matrix(op_h) 
bh = get_vector(op_h)
xh_exact = Ah\bh

AH = assemble_matrix(aH,UH,VH)
Mhh = assemble_matrix((u,v)->∫(u⋅v)*dΩh,Uh,Vh)

uh_exact = FEFunction(Uh,xh_exact)
l2_error(uh_exact,u_exact,dΩh)
l2_norm(∇⋅uh_exact,dΩh)

topo = get_grid_topology(model)
patches_mask = get_isboundary_face(topo,0)
PD = PatchDecomposition(model)
Ph = PatchFESpace(Vh,PD,reffe_u;conformity=H1Conformity())#,patches_mask)
Ωp = Triangulation(PD)
dΩp = Measure(Ωp,qdegree)
ap(x,y) = a(x,y,dΩp)
smoother = RichardsonSmoother(PatchBasedLinearSolver(ap,Ph,Vh),20,0.01)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

function project_f2c(rh)
  Qrh = Mhh\rh
  uh = FEFunction(Vh,Qrh)
  ll(v) = ∫(v⋅uh)*dΩHh
  assemble_vector(ll,VH)
end
function interp_c2f(xH)
  get_free_dof_values(interpolate(FEFunction(VH,xH),Vh))
end

PDi = PatchBasedSmoothers.CoarsePatchDecomposition(model)
Ih = PatchFESpace(Vh,PDi,reffe_u;conformity=H1Conformity())

Ωpi = Triangulation(PDi)
dΩpi = Measure(Ωpi,qdegree)
api(x,y) = a(x,y,dΩpi)
Ai = assemble_matrix(api,Ih,Ih)

function P1(dxH)
  interp_c2f(dxH)
end
function R1(rh)
  project_f2c(rh)
end

function P2(dxH)
  dxh = interp_c2f(dxH)
  uh = FEFunction(Vh,dxh)
  r̃h = assemble_vector(v -> graddiv(uh,v,dΩpi),Ih)
  dx̃ = Ai\r̃h
  Pdxh = fill(0.0,length(dxh))
  PatchBasedSmoothers.inject!(Pdxh,Ih,dx̃)
  y = dxh - Pdxh
  return y
end
function R2(rh)
  r̃h = zero_free_values(Ih)
  PatchBasedSmoothers.prolongate!(r̃h,Ih,rh)
  dr̃h = Ai\r̃h
  dxh = zero_free_values(Vh)
  PatchBasedSmoothers.inject!(dxh,Ih,dr̃h)
  drh = assemble_vector(v -> graddiv(FEFunction(Vh,dxh),v,dΩh),Vh)
  rH = project_f2c(rh - drh)
  return rH
end


xh = randn(size(Ah,2))
rh = bh - Ah*xh
niters = 5

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

uh = FEFunction(Xh,xh)
l2_error(uh,u_exact,dΩh)
l2_norm(∇⋅uh,dΩh)
