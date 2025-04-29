# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-04-29

### Added

- Added support for GMG in serial. Since PR[#68](https://github.com/gridap/GridapSolvers.jl/pull/68).
- Added Vanka-like smoothers in serial. Since PR[#68](https://github.com/gridap/GridapSolvers.jl/pull/68).
- Added `StaggeredFEOperators` and `StaggeredFESolvers`. Since PR[#84](https://github.com/gridap/GridapSolvers.jl/pull/84).
- Added `RichardsonLinearSolver`. Since PR[#87](https://github.com/gridap/GridapSolvers.jl/pull/87).
- Added `NullspaceSolver` for serial. Since PR[#88](https://github.com/gridap/GridapSolvers.jl/pull/88).

### Changed

- **BREAKING:** Moved GridapP4est, GridapPETSc and IterativeSolvers into extensions (i.e weak dependencies). Since PR[#76](https://github.com/gridap/GridapSolvers.jl/pull/76).

## Previous versions

A changelog is not maintained for older versions than 0.4.0.
