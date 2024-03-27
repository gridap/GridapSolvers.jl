using Test
using BlockArrays, LinearAlgebra
using Gridap, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays, GridapDistributed
using GridapSolvers, GridapSolvers.BlockSolvers

function same_block_array(A,B)
  map(blocks(A),blocks(B)) do A, B
    t = map(partition(A),partition(B)) do A, B
      A ≈ B
    end
    reduce(&,t)
  end |> all
end

np = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(8,8))

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe)

mfs = BlockMultiFieldStyle()
Y = MultiFieldFESpace([V,V];style=mfs)

Ω  = Triangulation(model)
dΩ = Measure(Ω,4)

u0 = zero(Y)
sol(x) = sum(x)

# Reference operator
a_ref((u1,u2),(v1,v2)) = ∫(u1⋅v1 + u2⋅v2)*dΩ
l_ref((v1,v2)) = ∫(sol⋅v1 + sol⋅v2)*dΩ
res_ref(u,v) = a_ref(u,v) - l_ref(v)
jac_ref(u,du,dv) = a_ref(du,dv)
op_ref = FEOperator(res_ref,jac_ref,Y,Y)
A_ref = jacobian(op_ref,u0)
b_ref = residual(op_ref,u0)

# Block operator
a(u,v) = ∫(u⋅v)*dΩ
l(v)   = ∫(sol⋅v)*dΩ
res(u,v) = a(u,v) - l(v)
jac(u,du,dv) = a(du,dv)

res_blocks = collect(reshape([res,missing,missing,res],(2,2)))
jac_blocks = collect(reshape([jac,missing,missing,jac],(2,2)))
op = BlockFEOperator(res_blocks,jac_blocks,Y,Y)
A  = jacobian(op,u0)
b  = residual(op,u0)

@test same_block_array(A,A_ref)
@test same_block_array(b,b_ref)
