
using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers

using Gridap.MultiField

ranks = with_debug() do distribute
  distribute(LinearIndices((4,)))
end

np_per_level = [(2,2),(2,2),(2,1)]
nrefs = [(2,1),(2,2)]
isperiodic = (true,false)

mh = CartesianModelHierarchy(
  ranks,
  np_per_level,
  (0,1,0,1),
  (10,10);
  nrefs,
  isperiodic
)

ncells_per_level = map(num_cells∘get_model,mh).array

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(mh,reffe,dirichlet_tags="boundary")
X = MultiFieldFESpace([V,V])

PD = PatchDecomposition(mh)
Ph = PatchFESpace(V,PD)
