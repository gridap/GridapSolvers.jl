using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.Arrays
using Gridap.CellData

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

using PartitionedArrays, GridapDistributed

using LinearAlgebra


function DModel(parts,model)
  models = map(p -> model, parts)
  n = num_cells(model)
  indices = map(p -> LocalIndices(n,1,collect(1:n),fill(1,n)), parts)
  gids = PRange(indices)
  return GridapDistributed.DistributedDiscreteModel(models,gids)
end

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

fmodel = refine(cmodel)

np = 1
parts = with_mpi() do distribute
  distribute(LinearIndices((np,)))
end

dcmodel = DModel(parts,cmodel)
dfmodel = DModel(parts,fmodel)
dglue = map(p -> get_adaptivity_glue(fmodel), parts)
mh_clevel = MultilevelTools.ModelHierarchyLevel(2,dcmodel,nothing,nothing,nothing)
mh_flevel = MultilevelTools.ModelHierarchyLevel(1,dfmodel,dglue,nothing,nothing)
mh = HierarchicalArray([mh_flevel,mh_clevel],[parts,parts])

Ωh = Triangulation(dfmodel)
ΩH = Triangulation(dcmodel)

qdegree = 2*(order+1)
dΩh = Measure(Ωh,qdegree)
dΩH = Measure(ΩH,qdegree)
dΩHh = Measure(ΩH,Ωh,qdegree)

# Spaces
conformity = H1Conformity()
u_exact(x) = VectorValue(x[1]^2,-2.0*x[2]*x[1])
#u_exact(x) = VectorValue(x[1]*(x[1]-1.0)*x[2]*(x[2]-1.0),(1.0-2.0*x[1])*(1.0/3.0*x[2]^3 - 1.0/2.0*x[2]^2))

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
VH = TestFESpace(dcmodel,reffe,dirichlet_tags="boundary")
UH = TrialFESpace(VH,u_exact)
Vh = TestFESpace(dfmodel,reffe,dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u_exact)

# Weakform
α = 1.e10
f(x) = -Δ(u_exact)(x)
Π_Qh = LocalProjectionMap(divergence,lagrangian,Float64,order-1;space=:P)

lap(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ
graddiv(u,v,dΩ) = ∫(α*Π_Qh(v,dΩ)⋅Π_Qh(u,dΩ))dΩ
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

function project_f2c(rh)
  Qrh = Mhh\rh
  uh  = FEFunction(Vh,Qrh)
  assemble_vector(v->∫(v⋅uh)*dΩHh,VH)
end

# Smoother
PD = PatchDecomposition(dfmodel)
Ph = PatchFESpace(Vh,PD,reffe;conformity)
Ωp = Triangulation(PD)
dΩp = Measure(Ωp,qdegree)
ap(u,v) = biform(u,v,dΩp)
smoother = RichardsonSmoother(PatchBasedLinearSolver(ap,Ph,Vh),10,0.2)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)

# Prolongation Operator 1
Ṽh = FESpace(dfmodel,reffe;dirichlet_tags="coarse")
Ãh = assemble_matrix(ah,Ṽh,Ṽh)
function P1(dxH)
  uh = interpolate(FEFunction(VH,dxH),Vh)
  dxh = get_free_dof_values(uh)

  bh = assemble_vector(v -> graddiv(uh,v,dΩh),Ṽh)
  dx̃ = Ãh\bh
  ũh = interpolate(FEFunction(Ṽh,dx̃),Vh)

  y = dxh - get_free_dof_values(ũh)
  return y
end
function R1_bis(rh)
  r̃h = get_free_dof_values(interpolate(FEFunction(Vh,rh),Ṽh))
  dr̃h = Ãh\r̃h
  drh = get_free_dof_values(interpolate(FEFunction(Ṽh,dr̃h),Vh))
  rH = project_f2c(rh - drh)
  return rH
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
mh_Vh = FESpace(mh,reffe;dirichlet_tags="boundary")
cell_conformity = mh_Vh[1].cell_conformity
patches_mask = PatchBasedSmoothers.get_coarse_node_mask(dfmodel,dglue)
Ih = PatchFESpace(Vh,PD,cell_conformity;patches_mask=patches_mask)
I_solver = PatchBasedLinearSolver(ap,Ih,Vh)
I_ns = numerical_setup(symbolic_setup(I_solver,Ah),Ah)
Ai = assemble_matrix(ap,Ih,Ih)

function P2(dxH)
  uh = interpolate(FEFunction(VH,dxH),Vh)
  dxh = get_free_dof_values(uh)
  r̃h = assemble_vector(v -> graddiv(uh,v,dΩp),Ih)
  dx̃ = Ai\r̃h
  Pdxh = zero_free_values(Vh)
  PatchBasedSmoothers.inject!(Pdxh,Ih,dx̃)
  y = dxh - Pdxh
  return y
end
function R2_bis(rh)
  r̃h = zero_free_values(Ih)
  PatchBasedSmoothers.prolongate!(r̃h,Ih,rh)
  dr̃h = Ai\r̃h
  drh = zero_free_values(Vh)
  PatchBasedSmoothers.inject!(drh,Ih,dr̃h)
  rH = project_f2c(rh - drh)
  return rH
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

# Prolongation Operator 3

prolongations = setup_patch_prolongation_operators(
  mh_Vh,biform,graddiv,qdegree
);
restrictions = setup_patch_restriction_operators(
  mh_Vh,prolongations,graddiv,qdegree
);

function P3(dxH)
  dxh = zero_free_values(Vh)
  mul!(dxh,prolongations[1],dxH)
  return dxh
end
function R3(rh)
  rH = zero_free_values(UH)
  mul!(rH,restrictions[1],rh)
  return rH
end

# Solve
begin
xh = pfill(1.0,partition(axes(Ah,2)));
#xh = prandn(partition(axes(Ah,2)))
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

  rH = R3(rh)
  println("   > rH: ", norm(rH))
  qH = AH\rH
  println("   > qH: ", norm(qH))
  qh = P3(qH)
  println("   > qh: ", norm(qh))

  rh = rh - Ah*qh
  xh = xh + qh
  println(" > Post-correction: ", norm(rh))

  solve!(xh,smoother_ns,rh)

  iter += 1
  error = norm(rh)
  e_rel = error/error0
  println(" > Final: ",error, " - ", e_rel)
end
end
uh = FEFunction(Uh,xh)
eh = FEFunction(Vh,rh)
uh_star = FEFunction(Uh,xh_star)
