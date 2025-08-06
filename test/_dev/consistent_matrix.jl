
using Gridap
using GridapDistributed
using PartitionedArrays

using LinearAlgebra
using SparseArrays
using BlockArrays

using Gridap.Arrays

using GridapSolvers
using GridapSolvers.SolverInterfaces: fetch_ghost_rows

function mass(model)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  V = FESpace(model, reffe)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  a(u,v) = ∫(u⋅v)dΩ
  A = assemble_matrix(a, V, V)
  return A, V
end

function hdiv(model,order=1)
  reffe = ReferenceFE(raviart_thomas, Float64, order-1)
  V = FESpace(model, reffe)
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2*order)
  a(u,v) = ∫(u⋅v + (∇⋅u)*(∇⋅v))dΩ
  A = assemble_matrix(a, V, V)
  return A, V
end

parts = (2, 1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end
#ranks = collect(LinearIndices((prod(parts),)))

model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(8,8))

A, V = hdiv(model,2)

mats = partition(A)
rows = partition(axes(A,1))
cols = partition(axes(A,2))
@assert PartitionedArrays.matching_own_indices(PRange(rows), PRange(cols))

new_rows = map(partition(get_free_dof_ids(V))) do ids
  OwnAndGhostIndices(
    OwnIndices(global_length(ids), part_id(ids), own_to_global(ids)),
    GhostIndices(global_length(ids), ghost_to_global(ids), ghost_to_owner(ids))
  )
end

B = fetch_ghost_rows(A, new_rows)
new_rows, new_cols = partition(axes(B,1)), partition(axes(B,2));

A_main = PartitionedArrays.getany(partition(PartitionedArrays.to_trivial_partition(A)))

map(partition(B),partition(axes(B,1)),partition(axes(B,2))) do b, r, c
  b == A_main[r,c]
end

map(partition(B),partition(axes(B,1)),partition(axes(B,2))) do B, rows, cols
  r_o2l = own_to_local(rows)
  r_o2g = own_to_global(rows)
  r_g2l = ghost_to_local(rows)
  r_g2g = ghost_to_global(rows)

  c_o2l = own_to_local(cols)
  c_o2g = own_to_global(cols)
  c_g2l = ghost_to_local(cols)
  c_g2g = ghost_to_global(cols)

  B[r_o2l,c_o2l] == A_main[r_o2g,c_o2g] # Yes 
  B[r_o2l,c_g2l] == A_main[r_o2g,c_g2g] # Yes
  B[r_g2l,c_o2l] == A_main[r_g2g,c_o2g]
end
