# GridapPETSc.jl extension

```@meta
CurrentModule = Base.get_extension(GridapSolvers,:GridapPETScExt)
```

Building on top of [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl), GridapSolvers provides specific solvers for some particularly complex PDEs:

```@docs
  PETScElasticitySolver
  PETScElasticitySolver(::FESpace)
  CachedPETScNS
```
