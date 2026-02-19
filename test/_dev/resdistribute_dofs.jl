
using Gridap
using GridapDistributed
using PartitionedArrays
using GridapSolvers
using MPI

using Gridap.Arrays, Gridap.Geometry
using GridapSolvers.MultilevelTools

print_item(r, x) = println("   > Rank $r: $x")
function print_item(r, x::AbstractLocalIndices)
  println("   > Rank $r: ")
  println("      > no: $(own_length(x))")
  println("      > ng: $(ghost_length(x))")
  println("      > l2g: $(local_to_global(x))")
  println("      > l2o: $(local_to_owner(x))")
end

function mpi_print(x, name="")
  if !isnothing(x)
    ranks = linear_indices(x)
    nr = length(ranks)
    map(ranks, x) do r, x
      isa(x,DebugArray) || sleep(0.5*(r-1))
      if r == 1
        println("-------------------------------------------")
        println(" > $name : ")
      end
      print_item(r,x)
      if r == nr
        println("-------------------------------------------")
      end
    end
    PartitionedArrays.barrier(x)
  end
end

np = (2,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

ranks = DebugArray(LinearIndices((prod(np),)))
mh = CartesianModelHierarchy(ranks,[np,(1,1)],(0,1,0,1),(3,3);isperiodic=(false,true))

new_ranks = get_level_parts(mh,1)
old_ranks = get_level_parts(mh,2)

model_new = get_model(mh,1)
model_old = get_model_before_redist(mh,1)
glue = Base.unsafe_getindex(mh,1).red_glue

map(Geometry.num_vertices,local_views(model_old))
map(Geometry.num_vertices,local_views(model_new))

reffe = ReferenceFE(lagrangian, Float64, 1)

V_new = FESpace(model_new, reffe)
new_data_ids = partition(get_free_dof_ids(V_new))
new_cell_to_new_lid = map(get_cell_dof_ids,local_views(V_new))

if i_am_in(old_ranks)
  V_old = FESpace(model_old, reffe)
  old_data_ids = partition(get_free_dof_ids(V_old))
  old_cell_to_old_lid = map(get_cell_dof_ids,local_views(V_old))
else
  V_old, old_data_ids, old_cell_to_old_lid = nothing, nothing, nothing
end

num_free_dofs(V_new), num_free_dofs(V_old)

Ω_new = Triangulation(model_new)
dΩ_new = Measure(Ω_new, 2)
a_new(u,v) = ∫(u*v)dΩ_new
A_new = assemble_matrix(a_new, V_new, V_new)

if !isnothing(V_old)
  Ω_old = Triangulation(model_old)
  dΩ_old = Measure(Ω_old, 2)
  a_old(u,v) = ∫(u*v)dΩ_old
  A_old = assemble_matrix(a_old, V_old, V_old)
else
  A_old = nothing
end

############################################################################################

old_ids, red_old_ids = GridapDistributed.redistribute_indices(
  old_data_ids, old_cell_to_old_lid, new_cell_to_new_lid, model_new, glue; reverse=false,
);

new_ids, red_new_ids = GridapDistributed.redistribute_indices(
  new_data_ids, new_cell_to_new_lid, old_cell_to_old_lid, model_old, glue; reverse=true,
);

mpi_print(old_ids, "Old IDs")
mpi_print(red_old_ids, "Redistribute Old IDs")

mpi_print(new_ids, "New IDs")
mpi_print(red_new_ids, "Redistribute New IDs")

# Old -> New -> Old

x_old = consistent!(PVector(map(collect∘local_to_global,old_ids),old_ids)) |> fetch
x_old_red = redistribute(x_old, red_old_ids) |> fetch

x_new = PVector(partition(x_old_red), new_ids)
x_new_red = redistribute(x_new, red_new_ids) |> fetch

mpi_print(partition(x_old), "Partition Old")
mpi_print(partition(x_new_red), "Partition New Red")
display(map(≈,partition(x_old),partition(x_new_red)))

# New -> Old -> New

x_new = consistent!(PVector(map(collect∘local_to_global,new_ids),new_ids)) |> fetch
x_new_red = redistribute(x_new, red_new_ids) |> fetch

x_old = PVector(partition(x_new_red), old_ids)
x_old_red = redistribute(x_old, red_old_ids) |> fetch

mpi_print(partition(x_new), "Partition New")
mpi_print(partition(x_old_red), "Partition Old Red")
display(map(≈,partition(x_new),partition(x_old_red)))

#########################################################################################
# Matrix indices

function change_parts_indices(indices, new_parts)
  ng = !isnothing(indices) ? map(global_length,indices) : nothing
  l2g = !isnothing(indices) ? map(local_to_global, indices) : nothing
  l2o = !isnothing(indices) ? map(local_to_owner, indices) : nothing

  ng = emit(change_parts(ng,new_parts;default=0))
  new_l2g = change_parts(l2g, new_parts; default=Int[])
  new_l2o = change_parts(l2o, new_parts; default=Int32[])
  return map(LocalIndices,ng, new_parts, new_l2g, new_l2o)
end

function reindex_partition(new_indices, indices, reindexed_indices)
  l2g = PVector(map(collect∘local_to_global,reindexed_indices),indices)
  new_l2g = pzeros(Int, new_indices)
  t1 = consistent!(copy!(new_l2g, l2g))

  l2o = PVector(map(collect∘local_to_owner,reindexed_indices),indices)
  new_l2o = pzeros(Int32, new_indices)
  t2 = consistent!(copy!(new_l2o, l2o))

  wait(t1)
  wait(t2)

  reindexed_new_indices = map(reindexed_indices, partition(new_l2g), partition(new_l2o)) do reindexed_indices, new_l2g, new_l2o
    LocalIndices(global_length(reindexed_indices), part_id(reindexed_indices), new_l2g, new_l2o)
  end
  return reindexed_new_indices
end

new_ids_mat = partition(axes(A_new,2))
old_ids_mat = change_parts_indices(partition(axes(A_old,2)), new_ranks)

red_old_ids_mat = reindex_partition(new_ids_mat, new_ids, red_old_ids)
red_new_ids_mat = reindex_partition(old_ids_mat, old_ids, red_new_ids)

# Old -> New -> Old

x_old = consistent!(prandn(old_ids_mat)) |> fetch
x_old_red = redistribute(x_old, red_old_ids_mat) |> fetch

x_new = PVector(partition(x_old_red), new_ids_mat)
x_new_red = redistribute(x_new, red_new_ids_mat) |> fetch

mpi_print(partition(x_old), "Partition Old")
mpi_print(partition(x_new_red), "Partition New Red")
display(map(≈,partition(x_old),partition(x_new_red)))

# New -> Old -> New

x_new = consistent!(prandn(new_ids_mat)) |> fetch
x_new_red = redistribute(x_new, red_new_ids_mat) |> fetch

x_old = PVector(partition(x_new_red), old_ids_mat)
x_old_red = redistribute(x_old, red_old_ids_mat) |> fetch

mpi_print(partition(x_new), "Partition New")
mpi_print(partition(x_old_red), "Partition Old Red")
display(map(≈,partition(x_new),partition(x_old_red)))

#########################################################################################
# CoarsePatchTopology

mhl = mh[1]

model_new = get_model(mhl)
ptopo = GridapSolvers.PatchBasedSmoothers.CoarsePatchTopology(mhl)

cell_to_patch = map(
  (ptopo,model) -> map(x -> !isempty(x) ? Int(first(x)) : Int(0) ,Arrays.inverse_table(Geometry.get_patch_cells(ptopo), num_cells(model))),
  local_views(ptopo), local_views(model_new)
)
map(linear_indices(cell_to_patch),local_views(model_new), cell_to_patch) do r, model, c2p
  writevtk(Triangulation(model),"data/model_new_$r";celldata=["patches" => c2p])
end
