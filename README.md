# GridapSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapSolvers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapSolvers.jl/dev/)
[![Build Status](https://github.com/gridap/GridapSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gridap/GridapSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gridap/GridapSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gridap/GridapSolvers.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13327414.svg)](https://doi.org/10.5281/zenodo.13327414)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07162/status.svg)](https://doi.org/10.21105/joss.07162)

GridapSolvers provides algebraic and non-algebraic solvers for the Gridap ecosystem, designed with High Performance Computing (HPC) in mind.

Solvers follow a modular design, where most blocks can be combined to produce PDE-taylored solvers for a wide range of problems.

## (Non-exhaustive) list of features

- **Krylov solvers**: We provide a (short) list of Krylov solvers, with full preconditioner support and HPC-first implementation.
- **Block preconditioners**: We provide full support for block assembly of multiphysics problems, and a generic API for building block-based preconditioners for block-assembled systems.
- **Multilevel support**: We provide a generic API for building multilevel preconditioners.
- **Geometric Multigrid**: We provide a full-fledged geometric multigrid solver. Highly scalable adaptivity and redistribution of meshes, provided by `p4est` through `GridapP4est.jl`.
- **PETSc interface**: Full access to PETSc algebraic solvers, through `GridapPETSc.jl`, with full interoperability with the rest of the aforementioned solvers.

## Documentation and examples

A (hopefully) comprehensive documentation is available [here](https://gridap.github.io/GridapSolvers.jl/stable/).

A list of examples is available in the documentation. These include some very well known examples such as the Stokes, Incompressible Navier-Stokes and Darcy problems. The featured scripts are available in `test/Applications`.

An example on how to use the library within an HPC cluster is available in `joss_paper/scalability`. The included example and drivers are used to generate the scalability results in our [JOSS paper](https://doi.org/10.21105/joss.07162).

## Installation

GridapSolvers is a registered package in the official [Julia package registry](https://github.com/JuliaRegistries/General).  Thus, the installation of GridapSolvers is straight forward using the [Julia's package manager](https://julialang.github.io/Pkg.jl/v1/). Open the Julia REPL, type `]` to enter package mode, and install as follows

```julia
pkg> add GridapSolvers
pkg> build
```

Building is required to link the external artifacts (e.g., PETSc, p4est) to the Julia environment. Restarting Julia is required after building in order to make the changes take effect.

### Using custom binaries

The previous installations steps will setup GridapSolvers to work using Julia's pre-compiled artifacts for MPI, PETSc and p4est. However, you can also link local copies of these libraries. This might be very desirable in clusters, where hardware-specific libraries might be faster/more stable than the ones provided by Julia. To do so, follow the next steps:

- [MPI.jl](https://juliaparallel.org/MPI.jl/stable/configuration/)
- [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl)
- [GridapP4est.jl](https://github.com/gridap/GridapP4est.jl), and [P4est_wrapper.jl](https://github.com/gridap/p4est_wrapper.jl)

## Citation

In order to give credit to the `GridapSolvers` contributors, we simply ask you to cite the `Gridap` main project as indicated [here](https://github.com/gridap/Gridap.jl#how-to-cite-gridap) and the sub-packages you use as indicated in the corresponding repositories. Please, use the reference below in any publication in which you have made use of `GridapSolvers`:

```
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

## Contributing

GridapSolvers is a collaborative project open to contributions. If you want to contribute, please take into account:

  - Before opening a PR with a significant contribution, contact the project administrators by [opening an issue](https://github.com/gridap/GridapSolvers.jl/issues/new) describing what you are willing to implement. Wait for feedback from other community members.
  - We adhere to the contribution and code-of-conduct instructions of the Gridap.jl project, available [here](https://github.com/gridap/Gridap.jl/blob/master/CONTRIBUTING.md) and [here](https://github.com/gridap/Gridap.jl/blob/master/CODE_OF_CONDUCT.md), resp.  Please, carefully read and follow the instructions in these files.
  - Open a PR with your contribution.

Want to help? We have [issues waiting for help](https://github.com/gridap/GridapSolvers.jl/labels/help%20wanted). You can start contributing to the GridapSolvers project by solving some of those issues.
