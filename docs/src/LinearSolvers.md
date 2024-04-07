
```@meta
CurrentModule = GridapSolvers.LinearSolvers
```

# GridapSolvers.LinearSolvers

## Krylov solvers

```@docs
  CGSolver
  MINRESSolver
  GMRESSolver
  FGMRESSolver
  krylov_mul!
  krylov_residual!
```

## Smoothers

```@docs
  RichardsonSmoother
```

## Preconditioners

```@docs
  JacobiLinearSolver
```

## Wrappers

### PETSc

Building on top of [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl), GridapSolvers provides specific solvers for some particularly complex PDEs:

```@docs
  ElasticitySolver
  CachedPETScNS
  get_dof_coordinates
```

### IterativeSolvers.jl

GridapSolvers provides wrappers for some iterative solvers from the package [IterativeSolvers.jl](https://iterativesolvers.julialinearalgebra.org/dev/):

```@docs
  IterativeLinearSolver
  IS_ConjugateGradientSolver
  IS_GMRESSolver
  IS_MINRESSolver
  IS_SSORSolver
```
