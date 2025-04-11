# GridapSolvers

```@meta
CurrentModule = GridapSolvers
```

Documentation for [GridapSolvers](https://github.com/gridap/GridapSolvers.jl).

GridapSolvers provides algebraic and non-algebraic solvers for the Gridap ecosystem, designed with High Performance Computing (HPC) in mind.

Solvers follow a modular design, where most blocks can be combined to produce PDE-taylored solvers for a wide range of problems.

## Table of contents

```@contents
Pages = [
  "SolverInterfaces.md",
  "MultilevelTools.md",
  "LinearSolvers.md",
  "NonlinearSolvers.md",
  "BlockSolvers.md",
  "PatchBasedSmoothers.md",
]
```

## Installation

GridapSolvers is a registered package in the official [Julia package registry](https://github.com/JuliaRegistries/General).  Thus, the installation of GridapSolvers is straight forward using the [Julia's package manager](https://julialang.github.io/Pkg.jl/v1/). Open the Julia REPL, type `]` to enter package mode, and install as follows

```julia
pkg> add GridapSolvers
pkg> build
```

If using the extensions for `GridapP4est.jl` or `GridapPETSc.jl`, building is required to link the external artifacts (e.g., PETSc, p4est) to the Julia environment. Restarting Julia is required after building in order to make the changes take effect.

By default, Julia will configure `GridapSolvers` to work using Julia's pre-compiled artifacts for MPI, PETSc and p4est. However, you can also link local copies of these libraries. This might be very desirable in clusters, where hardware-specific libraries might be faster/more stable than the ones provided by Julia. To do so, follow the next steps:

- [MPI.jl](https://juliaparallel.org/MPI.jl/stable/configuration/)
- [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl)
- [GridapP4est.jl](https://github.com/gridap/GridapP4est.jl), and [P4est_wrapper.jl](https://github.com/gridap/p4est_wrapper.jl)

## Citation

In order to give credit to the `GridapSolvers` contributors, we simply ask you to cite the `Gridap` main project as indicated [here](https://github.com/gridap/Gridap.jl#how-to-cite-gridap) and the sub-packages you use as indicated in the corresponding repositories. Please, use the reference below in any publication in which you have made use of `GridapSolvers`:

```latex
@article{Manyer2024, 
  doi = {10.21105/joss.07162}, 
  url = {https://doi.org/10.21105/joss.07162}, 
  year = {2024}, 
  publisher = {The Open Journal}, 
  volume = {9}, 
  number = {102}, 
  pages = {7162}, 
  author = {Jordi Manyer and Alberto F. Martín and Santiago Badia}, 
  title = {GridapSolvers.jl: Scalable multiphysics finite element solvers in Julia}, 
  journal = {Journal of Open Source Software} 
} 
```
