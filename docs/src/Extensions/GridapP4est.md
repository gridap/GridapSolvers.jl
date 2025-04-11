# GridapP4est.jl extension

```@meta
CurrentModule = Base.get_extension(GridapSolvers,:GridapP4estExt)
```

Building on top of [GridapP4est.jl](https://github.com/gridap/GridapP4est.jl), GridapSolvers provides tools to use the P4est library to build `DiscreteModelHierarchy` objects.

```@docs
P4estCartesianModelHierarchy
```

## Examples of usage

```julia
# Start from a coarse mesh, then refine
function p4est_mesh_by_refinement(distribute,np_per_level,domain,nc;nrefs=0)
  GridapP4est.with(parts) do
    parts  = distribute(LinearIndices((np_per_level[1],)))
    cparts = generate_subparts(parts,np_per_level[end])
    base   = CartesianDiscreteModel(domain,nc)
    cmodel = OctreeDistributedDiscreteModel(cparts,base,nrefs)
    return ModelHierarchy(parts,cmodel,np_per_level)
  end
end

# Start from a fine mesh, then coarsen
function p4est_mesh_by_coarsening(distribute,np_per_level,domain,nc;nrefs=0)
  GridapP4est.with(parts) do
    n_levs = length(np_per_level)
    fparts = distribute(LinearIndices((np_per_level[1],)))
    base   = CartesianDiscreteModel(domain,nc)
    fmodel = OctreeDistributedDiscreteModel(fparts,base,nrefs+n_levs)
    return ModelHierarchy(parts,fmodel,np_per_level)
  end
end
```
