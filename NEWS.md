# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-08-30

### Breaking

- Complete rework of patch-based smoothers. Since PR[#94](https://github.com/gridap/GridapSolvers.jl/pull/94/files). List of breaking changes:
  - All the patch machinery has been removed (`PatchDecomposition`, `PatchFESpace`, `PatchTriangulation`, etc...) in favour of the new Gridap patch machinery (since Gridap v0.19).
  - The old `PatchBasedLinearSolver` has been renamed to `PatchSolver` and now uses the new patch machinery. The constructors are similar but should be simpler and more flexible/robust. This still integrates the blocks on demand.
  - The old `VankaSolver` has been renamed to `BlockJacobiSolver` and has been extended to distributed. It it now an alternative to `PatchSolver` where the local problems are extracted directly from the matrix instead of re-computing the local problems (and are equivalent if the patched weakform coincides with the original problem).
  - `PatchProlongationOperator` now uses the new `PatchSolver` machinery. A new operator `BlockJacobiProlongationOperator` has been added, which is equivalent but uses the `BlockJacobiSolver` machinery.

### Added

- `GMGLinearSolver` now supports F- and W-cycles on top of the existing V-cycle. The iteration type can be now chosen through the `cycle_type` kwarg. Since PR[#94](https://github.com/gridap/GridapSolvers.jl/pull/94/files).

## [0.6.1] - 2025-07-25

### Added

- Added support for the `HPDDM` library, with a new solver `HPDDMLinearSolver`. Since PR[#95](https://github.com/gridap/GridapSolvers.jl/pull/95).

## [0.6.0] - 2025-06-13

### Breaking

- Removed the `ModelHierarchy` from the `GMGLinearSolver` constructors. This should allow more flexibility for the GMG solver (for instance, to create a P-GMG). Since PR[#92](https://github.com/gridap/GridapSolvers.jl/pull/92).

### Added

- Added support for Gridap v0.19. Since PR[#92](https://github.com/gridap/GridapSolvers.jl/pull/92).
- Added a new extension for `Pardiso.jl`. Since PR[#92](https://github.com/gridap/GridapSolvers.jl/pull/92).

## [0.5.0] - 2025-04-29

### Breaking

- Moved dependencies for GridapP4est, GridapPETSc and IterativeSolvers into extensions (i.e weak dependencies). Since PR[#76](https://github.com/gridap/GridapSolvers.jl/pull/76).

### Added

- Added support for GMG in serial. Since PR[#68](https://github.com/gridap/GridapSolvers.jl/pull/68).
- Added Vanka-like smoothers in serial. Since PR[#68](https://github.com/gridap/GridapSolvers.jl/pull/68).
- Added `StaggeredFEOperators` and `StaggeredFESolvers`. Since PR[#84](https://github.com/gridap/GridapSolvers.jl/pull/84).
- Added `RichardsonLinearSolver`. Since PR[#87](https://github.com/gridap/GridapSolvers.jl/pull/87).
- Added `NullspaceSolver` for serial. Since PR[#88](https://github.com/gridap/GridapSolvers.jl/pull/88).

## Previous versions

A changelog is not maintained for older versions than 0.4.0.
