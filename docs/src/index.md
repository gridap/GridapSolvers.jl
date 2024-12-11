```@meta
CurrentModule = GridapSolvers
```

# GridapSolvers

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
  "PatchBasedSmoothers.md"
  ]
```

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
