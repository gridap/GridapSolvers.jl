
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

## Richardson iterative solver

Given a linear system ``Ax = b`` and a left **preconditioner** ``Pl``, this iterative solver performs the following iteration until the solution converges.

```math
  x_{k+1} = x_k + \omega Pl^{-1} (b - A x_k)
```

```@docs
  RichardsonLinearSolver
```

## Smoothers

Given a linear system ``Ax = b``, a **smoother** is an operator `S` that takes an iterative solution ``x_k`` and its residual ``r_k = b - A x_k``, and modifies them **in place**

```math
  S : (x_k,r_k) \rightarrow (x_{k+1},r_{k+1})
```

such that ``|r_{k+1}| < |r_k|``.

```@docs
  RichardsonSmoother
  RichardsonSmoother(M::LinearSolver)
```

## Preconditioners

Given a linear system ``Ax = b``, a **preconditioner** is an operator that takes an iterative residual ``r_k`` and returns a correction ``dx_k``.

```@docs
  JacobiLinearSolver
  GMGLinearSolverFromMatrices
  GMGLinearSolverFromWeakform
  GMGLinearSolver
```

## Wrappers

### PETSc

Building on top of [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl), GridapSolvers provides specific solvers for some particularly complex PDEs:

```@docs
  ElasticitySolver
  ElasticitySolver(::FESpace)
  CachedPETScNS
  CachedPETScNS(::GridapPETSc.PETScLinearSolverNS,::AbstractVector,::AbstractVector)
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
