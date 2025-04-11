# GridapPETSc.jl extension

```@meta
CurrentModule = GridapSolvers
```

Building on top of [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl), GridapSolvers provides specific solvers for some particularly complex PDEs:

```@docs
  ElasticitySolver
  ElasticitySolver(::FESpace)
  CachedPETScNS
  CachedPETScNS(::GridapPETSc.PETScLinearSolverNS,::AbstractVector,::AbstractVector)
```
