
using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers

ranks = with_debug() do distribute
  distribute(LinearIndices((4,)))
end

np_per_level = [(2,2),(2,2),(2,1)]
nrefs = [(2,1),(2,2)]
isperiodic = (false,false)

mh = CartesianModelHierarchy(
  ranks,
  np_per_level,
  (0,1,0,1),
  (5,5);
  nrefs,
  isperiodic
)

ncells_per_level = map(num_cells∘get_model,mh).array
