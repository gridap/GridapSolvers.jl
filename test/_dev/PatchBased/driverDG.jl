using Gridap
using Gridap.CellData, Gridap.Geometry, Gridap.FESpaces, Gridap.Arrays, Gridap.Algebra

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

get_edge_measures(Ω::Triangulation,dΩ) = sqrt∘CellField(get_array(∫(1)dΩ),Ω)

function weakform_dg(u,v,Ω,Γ,Λ,qorder)
  dΩ = Measure(Ω, qorder)
  dΓ = Measure(Γ, qorder)
  dΛ = Measure(Λ, qorder)

  n_Γ = get_normal_vector(Γ)
  n_Λ = get_normal_vector(Λ)
  h_e_Λ = get_edge_measures(Λ,dΛ)
  h_e_Γ = get_edge_measures(Γ,dΓ)

  ω = 1000.0
  c = ∫(∇(u)⋅∇(v))*dΩ + 
      ∫(-jump(u⋅n_Λ)⋅mean(∇(v)) - mean(∇(u))⋅jump(v⋅n_Λ) + ω/h_e_Λ*jump(u⋅n_Λ)⋅jump(v⋅n_Λ))*dΛ +
      ∫(-(∇(u)⋅n_Γ)⋅v - u⋅(∇(v)⋅n_Γ) + ω/h_e_Γ*(u⋅n_Γ)⋅(v⋅n_Γ))*dΓ
  return c
end

function weakform(u,v,Ω,qorder)
  dΩ = Measure(Ω, qorder)
  c = ∫(∇(u)⋅∇(v))*dΩ
  return c
end

btags = ["boundary"]

model = CartesianDiscreteModel((0,1,0,1),(8,8))
PD = PatchDecomposition(model;boundary_tag_names=btags)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
Λ = SkeletonTriangulation(model)

Ωp = Triangulation(PD)
Γp = BoundaryTriangulation(PD)
Λp = SkeletonTriangulation(PD)

order = 2
qdegree = 2*(order+1)

dΩ = Measure(Ω, qdegree)
dΓ = Measure(Γ, qdegree)
dΛ = Measure(Λ, qdegree)

u_sol(x) = x[1]*(1-x[1])*x[2]*(1-x[2])
f(x) = Δ(u_sol)(x)
g = VectorValue(1.0,1.0)

# Continuous problem

reffe_h1 = ReferenceFE(lagrangian,Float64,order)
V_h1 = FESpace(model,reffe_h1;dirichlet_tags=btags)
P_h1 = PatchFESpace(V_h1,PD,reffe_h1)

a_h1(u,v) = weakform(u,v,Ω,qdegree)
l_h1(v) = ∫(f⋅v)*dΩ

op_h1 = AffineFEOperator(a_h1,l_h1,V_h1,V_h1)
A_h1, b_h1 = get_matrix(op_h1), get_vector(op_h1)

S_h1 = RichardsonSmoother(PatchBasedLinearSolver((u,v)->weakform(u,v,Ωp,qdegree),P_h1,V_h1),10,0.2)
solver_h1 = CGSolver(S_h1,verbose=true)
ns_h1 = numerical_setup(symbolic_setup(solver_h1,A_h1),A_h1)

x_h1 = allocate_in_domain(A_h1); fill!(x_h1,0.0)
solve!(x_h1,ns_h1,b_h1)

# Discontinuous problem

reffe_l2 = ReferenceFE(lagrangian,Float64,order)
V_l2 = FESpace(model,reffe_l2;conformity=:L2,dirichlet_tags="boundary")
P_l2 = PatchFESpace(V_l2,PD,reffe_l2;conformity=:L2)

a_l2(u,v) = weakform_dg(u,v,Ω,Γ,Λ,qdegree)
l_l2(v) = ∫(f⋅v)*dΩ

op_l2 = AffineFEOperator(a_l2,l_l2,V_l2,V_l2)
A_l2, b_l2 = get_matrix(op_l2), get_vector(op_l2)

S_l2 = RichardsonSmoother(PatchBasedLinearSolver((u,v)->weakform_dg(u,v,Ωp,Γp,Λp,qdegree),P_l2,V_l2),10,0.2)
solver_l2 = CGSolver(S_l2,verbose=true)
ns_l2 = numerical_setup(symbolic_setup(solver_l2,A_l2),A_l2)

x_l2 = allocate_in_domain(A_l2); fill!(x_l2,0.0)
solve!(x_l2,ns_l2,b_l2)


S_l2_ns = numerical_setup(symbolic_setup(S_l2,A_l2),A_l2)
solve!(x_l2,S_l2_ns,b_l2)


############################################################################################
### VECTOR POISSON PROBLEM



reffe_hdiv = ReferenceFE(raviart_thomas,Float64,order-1)
V_hdiv = FESpace(model,reffe_hdiv;dirichlet_tags="boundary")
P_hdiv = PatchFESpace(V_hdiv,PD,reffe_hdiv)

a_hdiv(u,v) = weakform_dg(u,v,Ω,Γ,Λ,qdegree)
l_hdiv(v) = ∫(f⋅v)*dΩ

op = AffineFEOperator(a_hdiv,l_hdiv,V_hdiv,V_hdiv)
A_hdiv, b_hdiv = get_matrix(op), get_vector(op)

S_hdiv = RichardsonSmoother(PatchBasedLinearSolver((u,v)->weakform_dg(u,v,Ωp,Γp,Λp,qdegree),P_hdiv,V_hdiv),10,0.2)
solver_hdiv = CGSolver(S_hdiv,verbose=true)
ns_hdiv = numerical_setup(symbolic_setup(solver_hdiv,A_hdiv),A_hdiv)

x_hdiv = allocate_in_domain(A_hdiv); fill!(x,0.0)
solve!(x_hdiv,ns_hdiv,b_hdiv)

