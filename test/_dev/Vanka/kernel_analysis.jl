
using Gridap
using GridapSolvers
using LinearAlgebra

using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Geometry
using Gridap.Algebra, Gridap.Arrays
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools

Dc = 2
model = CartesianDiscreteModel((0,1,0,1),(4,4))

order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

V = TestFESpace(model,reffe_u)
Q = TestFESpace(model,reffe_p,conformity=:L2)
X = MultiFieldFESpace([V,Q])

qdegree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

PD = PatchDecomposition(model)
PD.patch_cells 

ν = 1.e-8
η = -1.0
ε = 1.e-4
Π = LocalProjectionMap(divergence,reffe_p)
#graddiv(u,v,dΩ) = ∫(η*(∇⋅v)⋅Π(u))*dΩ
graddiv(u,v,dΩ) = ∫(η*Π(v)⋅Π(u))*dΩ
lap(u,v,dΩ) = ∫(ν*∇(u)⊙∇(v))*dΩ
mass(u,v,dΩ) = ∫(u*v)*dΩ

function a((u,p),(v,q),dΩ)
  c = lap(u,v,dΩ)
  c += ∫((∇⋅v)*p + (∇⋅u)*q)dΩ
  if η > 0.0
    c += graddiv(u,v,dΩ)
  end
  if ε > 0.0
    c += mass(p,q,dΩ)
  end
  return c
end

A = assemble_matrix((u,v) -> a(u,v,dΩ),X,X)
x = A\ones(size(A,1))

cells = [6,7,10,11]
dof_ids = get_cell_dof_ids(X)
patch_ids = unique(sort(vcat(map(ids -> vcat(ids.array...), dof_ids[cells])...)))

A_vanka = Matrix(A[patch_ids,patch_ids])
cond(A_vanka)
x_vanka = A_vanka\randn(size(A_vanka,1))

s = VankaSolver(X,PD)
ns = numerical_setup(symbolic_setup(s,A),A)

for i in 1:10
  x = zeros(size(A,1))
  x_exact = randn(size(A,1))
  b = A*x_exact
  solve!(x,ns,b)
  println(norm(x-x_exact))
end
