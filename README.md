# GridapSolvers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapSolvers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapSolvers.jl/dev/)
[![Build Status](https://github.com/gridap/GridapSolvers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gridap/GridapSolvers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/gridap/GridapSolvers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/gridap/GridapSolvers.jl)

GridapSolvers provides algebraic and non-algebraic solvers for the Gridap ecosystem, designed with High Performance Computing (HPC) in mind.

Solvers follow a modular design, where most blocks can be combined to produce PDE-taylored solvers for a wide range of problems.

## (Non-exhaustive) list of features

- **Krylov solvers**: We provide a (short) list of Krylov solvers, with full preconditioner support and HPC-first implementation.
- **Block preconditioners**: We provide full support for block assembly of multiphysics problems, and a generic API for building block-based preconditioners for block-assembled systems.
- **Multilevel support**: We provide a generic API for building multilevel preconditioners.
- **Geometric Multigrid**: We provide a full-fledged geometric multigrid solver. Highly scalable adaptivity and redistribution of meshes, provided by `p4est` through `GridapP4est.jl`.
- **PETSc interface**: Full access to PETSc algebraic solvers, through `GridapPETSc.jl`, with full interoperability with the rest of the aforementioned solvers.

## Installation

GridapSolvers is a registered package in the official [Julia package registry](https://github.com/JuliaRegistries/General).  Thus, the installation of Gridap is straight forward using the [Julia's package manager](https://julialang.github.io/Pkg.jl/v1/). Open the Julia REPL, type `]` to enter package mode, and install as follows

```julia
pkg> add GridapSolvers
pkg> build
```

Building is required to link the external artifacts (e.g., PETSc, p4est) to the Julia environment.

### Using custom binaries

The previous installations steps will setup GridapSolvers to work using Julia's pre-compiled artifacts for MPI, PETSc and p4est. However, you can also link local copies of these libraries. This might be very desirable in clusters, where hardware-specific libraries might be faster/more stable than the ones provided by Julia. To do so, follow the next steps:

- [MPI.jl](https://juliaparallel.org/MPI.jl/stable/configuration/)
- [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl)
- [GridapP4est.jl](https://github.com/gridap/GridapP4est.jl), and [P4est_wrapper.jl](https://github.com/gridap/p4est_wrapper.jl)
