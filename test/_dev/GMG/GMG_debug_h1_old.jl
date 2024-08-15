using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.Arrays
using Gridap.CellData, Gridap.Fields

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

using LinearAlgebra

order = 3
poly  = QUAD

# Geometry 
n = 6
cmodel = CartesianDiscreteModel((0,1,0,1),(n,n))
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

# Spaces
conformity = H1Conformity()
u_exact(x) = VectorValue(x[1]^2,-2.0*x[2]*x[1])
#u_exact(x) = VectorValue(x[1]*(x[1]-1.0)*x[2]*(x[2]-1.0),(1.0-2.0*x[1])*(1.0/3.0*x[2]^3 - 1.0/2.0*x[2]^2))

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
VH = TestFESpace(cmodel,reffe,dirichlet_tags="boundary")
UH = TrialFESpace(VH,u_exact)
Vh = TestFESpace(fmodel,reffe,dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u_exact)

# Weakform
α = 1.e8
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
smoother = RichardsonSmoother(PatchBasedLinearSolver(ap,Ph,Vh),20,0.2)
smoother_ns = numerical_setup(symbolic_setup(smoother,Ah),Ah)


# New prolongation operator
ftopo = get_grid_topology(fmodel)
n2e_map = Gridap.Geometry.get_faces(ftopo,0,1)
e2n_map = Gridap.Geometry.get_faces(ftopo,1,0)
ccoords = get_node_coordinates(cmodel)
fcoords = get_node_coordinates(fmodel)

function is_fine(n)
  A = fcoords[n] ∉ ccoords
  edges = n2e_map[n]
  for e in edges
    nbor_nodes = e2n_map[e]
    A = A && all(m -> fcoords[m] ∉ ccoords,nbor_nodes)
  end
  return !A
end

_patches_mask = map(is_fine,LinearIndices(fcoords))
patches_mask = reshape(_patches_mask,length(fcoords))
Ih = PatchFESpace(Vh,PD,reffe;conformity=conformity,patches_mask=patches_mask)
I_solver = PatchBasedLinearSolver(ap,Ih,Vh)
I_ns = numerical_setup(symbolic_setup(I_solver,Ah),Ah)

Ai = assemble_matrix(ap,Ih,Ih)


patches_mask_2 = GridapSolvers.PatchBasedSmoothers.get_coarse_node_mask(fmodel,fmodel.glue)
patches_mask_2 == patches_mask
_patches_mask_2 = reshape(patches_mask_2,size(fcoords))


function prolongate(dxH)
  dxh = interp_c2f(dxH)
  uh = FEFunction(Vh,dxh)

  bh = assemble_vector(v -> graddiv(uh,v,dΩp),Ih)
  dx̃ = Ai\bh

  Pdxh = fill(0.0,length(dxh))
  GridapSolvers.PatchBasedSmoothers.inject!(Pdxh,Ih,dx̃)
  y = dxh - Pdxh

  return y
end

# Solve

xh = fill(1.0,size(Ah,2));
rh = bh - Ah*xh
niters = 20

iter = 0
error = norm(bh - Ah*xh)
while iter < niters && error > 1.0e-8
  println("Iter $iter:")
  println(" > Initial: ", norm(rh))

  solve!(xh,smoother_ns,rh)

  println(" > Pre-smoother: ", norm(rh))

  rH = project_f2c(rh)
  qH = AH\rH
  qh = prolongate(qH)

  rh = rh - Ah*qh
  xh = xh + qh
  println(" > Post-correction: ", norm(rh))

  solve!(xh,smoother_ns,rh)

  iter += 1
  error = norm(bh - Ah*xh)
  println(" > Final: ",error)
end

uh = FEFunction(Uh,xh)
eh = FEFunction(Vh,rh)
uh_star = FEFunction(Uh,xh_star)
#writevtk(Ωh,"data/h1div_error";cellfields=["eh"=>eh,"u"=>uh,"u_star"=>uh_star,"u_exact"=>u_exact])



"""
reffe_p = ReferenceFE(lagrangian,Float64,0;space=:P)
QH = FESpace(cmodel,reffe_p;conformity=:L2)

checks = fill(false,(num_free_dofs(QH),num_free_dofs(VH)))
for i in 1:num_free_dofs(QH)
  qH = zeros(num_free_dofs(QH)); qH[i] = 1.0
  for j in 1:num_free_dofs(VH)
    vH = zeros(num_free_dofs(VH)); vH[j] = 1.0
    vh = interp_c2f(vH)
    
    ϕH = FEFunction(QH,qH)
    φh = FEFunction(Vh,vh)
    φH = FEFunction(VH,vH)
    lhs = sum(∫(divergence(φh)*ϕH)*dΩh)
    rhs = sum(∫(divergence(φH)*ϕH)*dΩh)
    checks[i,j] = abs(lhs-rhs) < 1.0e-10
  end
end
all(checks)

reffe = LagrangianRefFE(VectorValue{2,Float64},QUAD,2)
dof_ids = get_cell_dof_ids(Vh)
local_dof_nodes = lazy_map(Reindex(reffe.reffe.dofs.nodes),reffe.reffe.dofs.dof_to_node)
cell_maps = get_cell_map(fmodel)
dof_nodes = Vector{VectorValue{2,Float64}}(undef,num_free_dofs(Vh))
for (ids,cmap) in zip(dof_ids,cell_maps)
  for (i,id) in enumerate(ids)
    if id > 0
      dof_nodes[id] = cmap(local_dof_nodes[i])
    end
  end
end

V̂h_dofs = findall(x -> !isempty(x),Ih.dof_to_pdof)
checks = fill(false,(num_free_dofs(QH),length(V̂h_dofs)))
for i in 1:num_free_dofs(QH)
  qH = zeros(num_free_dofs(QH)); qH[i] = 1.0
  for (j,j_dof) in enumerate(V̂h_dofs)
    vh = zeros(num_free_dofs(Vh)); vh[j_dof] = 1.0
    
    ϕH = FEFunction(QH,qH)
    φh = FEFunction(Vh,vh)
    lhs = sum(∫(divergence(φh)*ϕH)*dΩHh)
    checks[i,j] = abs(lhs) < 1.0e-10
  end
end
all(checks)
"""