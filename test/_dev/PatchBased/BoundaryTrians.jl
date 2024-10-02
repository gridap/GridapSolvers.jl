
using Test
using LinearAlgebra
using FillArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.FESpaces, Gridap.Geometry
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers

function get_cell_size(Ω::Triangulation{Dc}) where Dc
  dΩ = Measure(Ω,1)
  h_e = Gridap.CellData.get_array(∫(1)dΩ) .^ (1/Dc)
  return CellField(h_e,Ω)
end

function weakforms(model)
  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)
  Λ = SkeletonTriangulation(model)
  
  dΩ = Measure(Ω, qorder) 
  dΓ = Measure(Γ, qorder)
  dΛ = Measure(Λ, qorder)

  n_Γ = get_normal_vector(Γ)
  n_Λ = get_normal_vector(Λ)

  h_e = get_cell_size(Λ)

  a1(u,v) = ∫(∇(u)⊙∇(v))dΩ
  a2(u,v) = ∫((∇(v)⋅n_Γ)⋅u)dΓ
  a3(u,v) = ∫(jump(u⋅n_Λ)⋅jump(v⋅n_Λ))dΛ
  a4(u,v) = ∫(h_e*jump(u⋅n_Λ)⋅jump(v⋅n_Λ))dΛ
  return a1, a2, a3, a4
end

model = CartesianDiscreteModel((0,1,0,1),(2,2))
model = Gridap.Adaptivity.refine(model)

#order = 1
#qorder = 2*order+1
#reffe = ReferenceFE(raviart_thomas,Float64,order-1)
#Vh = FESpace(model,reffe)

order = 2
qorder = 2*order+1
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
Vh = FESpace(model,reffe)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)
Λ = SkeletonTriangulation(model)

PD = PatchDecomposition(model)
Ph = PatchFESpace(Vh,PD,reffe)

Ωp = Triangulation(PD)
Γp = BoundaryTriangulation(PD)
Λp = SkeletonTriangulation(PD)

Γp_virtual = BoundaryTriangulation(PD,tags=["boundary","interior"],reverse=true)

a1, a2, a3, a4 = weakforms(model)
ap1, ap2, ap3, ap4 = weakforms(PD)

A1 = assemble_matrix(a1,Vh,Vh)
Ap1 = assemble_matrix(ap1,Ph,Ph)

A2 = assemble_matrix(a2,Vh,Vh)
Ap2 = assemble_matrix(ap2,Ph,Ph)

A3 = assemble_matrix(a3,Vh,Vh)
Ap3 = assemble_matrix(ap3,Ph,Ph)

A4 = assemble_matrix(a4,Vh,Vh)
Ap4 = assemble_matrix(ap4,Ph,Ph)


Λ = SkeletonTriangulation(PD)
dΛ = Measure(Λ, qorder)
n_Λ = get_normal_vector(Λ)

up = get_trial_fe_basis(Ph)
ap(u,v) = ∫(jump(u⋅n_Λ)⋅jump(v⋅n_Λ))dΛ

A = assemble_matrix(ap,Ph,Ph)


@which is_change_possible(Ω,Λ)
