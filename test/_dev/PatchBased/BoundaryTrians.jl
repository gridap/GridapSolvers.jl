
using Test
using LinearAlgebra
using FillArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.FESpaces
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

function weakforms(model)
  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)
  Λ = SkeletonTriangulation(model)
  
  dΩ = Measure(Ω, qorder) 
  dΓ = Measure(Γ, qorder)
  dΛ = Measure(Λ, qorder)

  n_Γ = get_normal_vector(Γ)
  n_Λ = get_normal_vector(Λ)

  a1(u,v) = ∫(∇(u)⊙∇(v))dΩ
  a2(u,v) = ∫(∇(v)⋅n_Γ⋅u)dΓ
  a3(u,v) = ∫(jump(u⋅n_Λ)⋅jump(v⋅n_Λ))dΛ
  return a1, a2, a3
end

model = CartesianDiscreteModel((0,1,0,1),(2,2))

order = 1
qorder = 2*order+1
reffe = ReferenceFE(raviart_thomas,Float64,order-1)
Vh = FESpace(model,reffe)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
Λ = SkeletonTriangulation(model)

PD = PatchDecomposition(model)
Ph = PatchFESpace(Vh,PD,reffe)

Ωp = Triangulation(PD)
Γp = BoundaryTriangulation(PD)
Λp = SkeletonTriangulation(PD)

a1, a2, a3 = weakforms(model)
ap1, ap2, ap3 = weakforms(PD)

A1 = assemble_matrix(a1,Vh,Vh)
Ap1 = assemble_matrix(ap1,Ph,Ph)

A2 = assemble_matrix(a2,Vh,Vh)
Ap2 = assemble_matrix(ap2,Ph,Ph)

A3 = assemble_matrix(a3,Vh,Vh)
Ap3 = assemble_matrix(ap3,Ph,Ph)

u = get_trial_fe_basis(Ph)
v = get_fe_basis(Ph)

cm1 = collect_cell_matrix(Ph,Ph,ap1(u,v))
cm2 = collect_cell_matrix(Ph,Ph,ap2(u,v))
cm3 = collect_cell_matrix(Ph,Ph,ap3(u,v))
