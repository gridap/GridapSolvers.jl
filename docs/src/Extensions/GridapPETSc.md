# GridapPETSc.jl extension

```@meta
CurrentModule = Base.get_extension(GridapSolvers,:GridapPETScExt)
```

Building on top of [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl), GridapSolvers provides examples of complex solvers one can build.

## Elasticity solver

```@docs
  PETScElasticitySolver
  PETScElasticitySolver(::FESpace)
```

An example on how to use it on an elasticity problem can be found in ['test/Applications/Elasticity.jl'](https://github.com/gridap/GridapSolvers.jl/tree/main/test/Applications/Elasticity.jl).

## HPDDM solver

We also provide support for the [HPDDM library](https://github.com/hpddm/hpddm) through PETSc's [`PCHPDDM` preconditioner](https://petsc.org/main/manualpages/PC/PCHPDDM/):

```@docs
  HPDDMLinearSolver
```

An example on how to use it on a Poisson problem can be found in ['test/ExtLibraries/drivers/HPDDMTests.jl'](https://github.com/gridap/GridapSolvers.jl/tree/main/test/ExtLibraries/drivers/HPDDMTests.jl).

## Caching PETSc solvers

To have zero allocations when solving linear systems, one needs to pre-allocate PETSc arrays for the solution and right-hand side. We provide a way to automate this process:

```@docs
  CachedPETScNS
```
