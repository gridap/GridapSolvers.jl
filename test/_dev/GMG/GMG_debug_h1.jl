using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.Arrays
using Gridap.CellData

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

using LinearAlgebra

order = 2
poly  = QUAD

# Geometry 
n = 6
cmodel = CartesianDiscreteModel((0,1,0,1),(n,n))
if poly == TRI
  cmodel = simplexify(cmodel)
end
labels = get_face_labeling(cmodel)

for D in 1:2
  for i in LinearIndices(labels.d_to_dface_to_entity[D])
    if labels.d_to_dface_to_entity[D][i] == 9 # Interior faces (not cells)
      labels.d_to_dface_to_entity[D][i] = 10 # new entity
    end
  end
end
push!(labels.tag_to_entities[9],10)
push!(labels.tag_to_entities,[1:8...,10])
push!(labels.tag_to_name,"coarse")

add_tag_from_tags!(labels,"top",[3,4,6])
add_tag_from_tags!(labels,"bottom",[1,2,5])
add_tag_from_tags!(labels,"walls",[7,8])

fmodel = refine(cmodel)

Ωh = Triangulation(fmodel)
ΩH = Triangulation(cmodel)

qdegree = 2*(order+1)
dΩh = Measure(Ωh,qdegree)
dΩH = Measure(ΩH,qdegree)
dΩHh = Measure(ΩH,Ωh,qdegree)

# Spaces
conformity = H1Conformity()
u_exact(x) = VectorValue(x[1]^2,-2.0*x[2]*x[1])
u_bottom = VectorValue(0.0,0.0)
u_top = VectorValue(1.0,0.0)

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
#VH = TestFESpace(cmodel,reffe,dirichlet_tags="boundary")
#UH = TrialFESpace(VH,u_exact)
#Vh = TestFESpace(fmodel,reffe,dirichlet_tags="boundary")
#Uh = TrialFESpace(Vh,u_exact)
VH = TestFESpace(cmodel,reffe,dirichlet_tags=["bottom","top"])
UH = TrialFESpace(VH,[u_bottom,u_top])
Vh = TestFESpace(fmodel,reffe,dirichlet_tags=["bottom","top"])
Uh = TrialFESpace(Vh,[u_bottom,u_top])

# Weakform
α = 1.e10
f(x) = -Δ(u_exact)(x)
Π_Qh = LocalProjectionMap(poly,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)

lap(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ
graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(v))⋅Π_Qh(divergence(u)))dΩ
biform(u,v,dΩ) = lap(u,v,dΩ) + graddiv(u,v,dΩ)
ah(u,v) = biform(u,v,dΩh)
aH(u,v) = biform(u,v,dΩH)
lh(v) = ∫(v⋅f)*dΩh
lH(v) = ∫(v⋅f)*dΩH

oph = AffineFEOperator(ah,lh,Uh,Vh)
opH = AffineFEOperator(aH,lH,UH,VH)

xh_star = get_free_dof_values(solve(oph))
xH_star = get_free_dof_values(solve(opH))

Ah, bh = get_matrix(oph), get_vector(oph);
AH, bH = get_matrix(opH), get_vector(opH);

Mhh = assemble_matrix((u,v)->∫(u⋅v)*dΩh,Vh,Vh)

function project_c2f(xH)
  uH = FEFunction(VH,xH)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩh,v->∫(v⋅uH)*dΩh,Vh,Vh)
  return get_matrix(op)\get_vector(op)
end

function project_f2c(rh)
  Qrh = Mhh\rh
  uh  = FEFunction(Vh,Qrh)
  assemble_vector(v->∫(v⋅uh)*dΩHh,VH)
end

function interp_c2f(xH)
  get_free_dof_values(interpolate(FEFunction(VH,xH),Vh))
end

# Smoother
PD = PatchDecomposition(fmodel)
Ph = PatchFESpace(Vh,PD,reffe;conformity)
Ωp = Triangulation(PD)
dΩp = Measure(Ωp,qdegree)
ap(u,v) = biform(u,v,dΩp)
smoother = RichardsonSmoother(PatchBasedLinearSolver(ap,Ph,Vh),10,0.2)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

# Prolongation Operator 1
Ṽh = FESpace(fmodel,reffe;dirichlet_tags="coarse")
Ãh = assemble_matrix(ah,Ṽh,Ṽh)
function P1(dxH)
  dxh = interp_c2f(dxH)
  uh = FEFunction(Vh,dxh)

  bh = assemble_vector(v -> graddiv(uh,v,dΩh),Ṽh)
  dx̃ = Ãh\bh
  ũh = interpolate(FEFunction(Ṽh,dx̃),Vh)

  y = dxh - get_free_dof_values(ũh)
  return y
end
function R1(rh)
  r̃h = get_free_dof_values(interpolate(FEFunction(Vh,rh),Ṽh))
  dr̃h = Ãh\r̃h
  dxh = interpolate(FEFunction(Ṽh,dr̃h),Vh)
  drh = assemble_vector(v -> graddiv(dxh,v,dΩh),Vh)
  rH = project_f2c(rh - drh)
  return rH
end

# Prolongation Operator 2

patches_mask = PatchBasedSmoothers.get_coarse_node_mask(fmodel,fmodel.glue)
Ih = PatchFESpace(Vh,PD,reffe;conformity=conformity,patches_mask=patches_mask)
I_solver = PatchBasedLinearSolver(ap,Ih,Vh)
I_ns = numerical_setup(symbolic_setup(I_solver,Ah),Ah)
Ai = assemble_matrix(ap,Ih,Ih)

function P2(dxH)
  dxh = interp_c2f(dxH)
  uh = FEFunction(Vh,dxh)
  r̃h = assemble_vector(v -> graddiv(uh,v,dΩp),Ih)
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

# Solve

#xh = fill(1.0,size(Ah,2));
xh = randn(size(Ah,2))
rh = bh - Ah*xh
niters = 100

iter = 0
error0 = norm(rh)
error = error0
e_rel = error/error0
while iter < niters && e_rel > 1.0e-10
  println("Iter $iter:")
  println(" > Initial: ", norm(rh))

  solve!(xh,smoother_ns,rh)
  println(" > Pre-smoother: ", norm(rh))

  rH = R1(rh)
  qH = AH\rH
  qh = P1(qH)

  rh = rh - Ah*qh
  xh = xh + qh
  println(" > Post-correction: ", norm(rh))

  solve!(xh,smoother_ns,rh)

  iter += 1
  error = norm(rh)
  e_rel = error/error0
  println(" > Final: ",error, " - ", e_rel)
end

uh = FEFunction(Uh,xh)
eh = uh - u_exact
uh_star = FEFunction(Uh,xh_star)

writevtk(Ωh,"data/solution",cellfields=["u_star"=>uh_star])
writevtk(cmodel,"data/cmodel")
#writevtk(Ωh,"data/solution",cellfields=["u"=>uh,"u_star"=>uh_star,"error"=>eh])
