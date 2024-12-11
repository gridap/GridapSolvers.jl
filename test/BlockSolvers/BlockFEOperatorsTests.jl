using Test
using BlockArrays, LinearAlgebra
using Gridap, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays, GridapDistributed
using GridapSolvers, GridapSolvers.BlockSolvers


model = CartesianDiscreteModel((0,1,0,1),(4,4))

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe)

Ω = Triangulation(model)
dΩ = Measure(Ω,3*order)
jac(u,du,dv) = ∫(u * du * dv)dΩ
res(u,dv) = ∫(u * dv)dΩ





