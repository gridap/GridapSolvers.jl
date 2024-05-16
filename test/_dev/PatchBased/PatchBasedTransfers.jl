
using Gridap
using GridapDistributed
using PartitionedArrays

using GridapSolvers
using GridapSolvers.MultilevelTools, GridapSolvers.PatchBasedSmoothers

np = 2
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

mh1 = CartesianModelHierarchy(parts,[np,np],(0,1,0,1),(2,2))
model1 = get_model(mh1,1)
glue1 = mh1[1].ref_glue

mh2 = CartesianModelHierarchy(parts,[np,1],(0,1,0,1),(2,2))
model2 = get_model_before_redist(mh2,1)
glue2 = mh2[1].ref_glue

gids1 = get_face_gids(model1,0)
mask1 = PatchBasedSmoothers.get_coarse_node_mask(model1,glue1)
display(local_to_global(gids1))
display(mask1)

if i_am_main(parts)
  gids2 = get_face_gids(model2,0)
  mask2 = PatchBasedSmoothers.get_coarse_node_mask(model2,glue2)
  display(local_to_global(gids2))
  display(mask2)
end
