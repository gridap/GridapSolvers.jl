
```@meta
CurrentModule = GridapSolvers.BlockSolvers
```

# GridapSolvers.BlockSolvers

Many scalable preconditioners for multiphysics problems are based on (possibly partial) block factorizations. This module provides a simple interface to define and use block solvers for block-assembled systems.

## Block types

In a same preconditioner, blocks can come from different sources. For example, in a Schur-complement-based preconditioner you might want to solve the eliminated block (which comes from the original matrix), while having an approximation for your Schur complement (which can come from a matrix assembled in your driver, or from a weakform).

For this reason, we define the following abstract interface:

```@docs
  SolverBlock
  LinearSolverBlock
  NonlinearSolverBlock
```

On top of this interface, we provide some useful block implementations:

```@docs
  LinearSystemBlock
  NonlinearSystemBlock
  MatrixBlock
  BiformBlock
  TriformBlock
```

To create a new type of block, one needs to implement the following implementation (similar to the one for `LinearSolver`):

```@docs
  block_symbolic_setup
  block_numerical_setup
  block_numerical_setup!
  block_offdiagonal_setup
  block_offdiagonal_setup!
```

## Block solvers

We can combine blocks to define a block solver. All block solvers take an array of blocks and a vector of solvers for the diagonal blocks (which need to be solved for). We provide two common types of block solvers:

### BlockDiagonalSolvers

```@docs
  BlockDiagonalSolver
  BlockDiagonalSolver(blocks::AbstractVector{<:SolverBlock},solvers::AbstractVector{<:LinearSolver})
  BlockDiagonalSolver(solvers::AbstractVector{<:LinearSolver})
  BlockDiagonalSolver(funcs::AbstractArray{<:Function},trials::AbstractArray{<:FESpace},tests::AbstractArray{<:FESpace},solvers::AbstractArray{<:LinearSolver})
```

### BlockTriangularSolvers

```@docs
BlockTriangularSolver
BlockTriangularSolver(blocks::AbstractMatrix{<:SolverBlock},solvers ::AbstractVector{<:LinearSolver},)
BlockTriangularSolver(solvers::AbstractVector{<:LinearSolver})
```

## Staggered FE Operators

```@docs
StaggeredFESolver
StaggeredFEOperator
AffineStaggeredFESolver
NonlinearStaggeredFESolver
solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB}, ::Nothing)
```
