---
title: 'GridapSolvers.jl: Scalable multiphysics finite element solvers in Julia'
tags:
  - Julia
  - pdes
  - finite elements
  - hpc
  - solvers
authors:
  - name: Jordi Manyer
    orcid: 0000-0002-0178-3890
    corresponding: true
    equal-contrib: true
    affiliation: "1"
  - name: Alberto Martín-Huertas
    orcid: 0000-0001-5751-4561
    equal-contrib: true
    affiliation: "2"
  - name: Santiago Badia
    orcid: 0000-0003-2391-4086
    corresponding: true
    affiliation: "1,3"
affiliations:
 - name: School of Mathematics, Monash University, Clayton, Victoria, 3800, Australia.
   index: 1
 - name: School of Computing, Australian National University, Autonomous territories of Canberra, Australia
   index: 2
 - name: Centre Internacional de Mètodes Numèrics en Enginyeria, Esteve Terrades 5, E-08860 Castelldefels, Spain.
   index: 3
date: XXX April 2024
bibliography: paper.bib

aas-journal: Journal of Open Source Software
---

# Summary and statement of need

The ever-increasing demand for resolution and accuracy in mathematical models of physical processes governed by systems of Partial Differential Equations (PDEs) can only be addressed using fully-parallel advanced numerical discretization methods and scalable solution methods, thus able to exploit the vast amount of computational resources in state-of-the-art supercomputers.

One of the biggest scalability bottlenecks within Finite Element (FE) parallel codes is the solution of linear systems arising from the discretization of PDEs.
The implementation of exact factorization-based solvers in parallel environments is an extremely challenging task, and even state-of-the-art libraries such as MUMPS [@MUMPS] or PARDISO [@PARDISO] have severe limitations in terms of scalability and memory consumption above a certain number of CPU cores.
Hence the use of iterative methods is crucial to maintain scalability of FE codes. Unfortunately, the convergence of iterative methods is not guaranteed and rapidly deteriorates as the size of the linear system increases. To retain performance, the use of highly scalable preconditioners is mandatory.
For most problems, algebraic solvers and preconditioners (i.e based uniquelly on the algebraic system) are enough to obtain robust convergence. Many well-known libraries providing algebraic solvers already exist, such as PETSc [@petsc-user-ref], Trilinos [@trilinos], or Hypre [@hypre]. However, algebraic solvers are not always suited to deal with some of the most challenging problems.
In these cases, geometric solvers (i.e., solvers that exploit the geometry and physics of the particular problem) are required. This is the case of many multiphysics problems, such as Navier-Stokes, Darcy or MHD. To this end, GridapSolvers is a registered Julia [@Bezanson2017] software package which provides highly scalable geometric solvers tailored for the FE numerical solution of PDEs on parallel computers.

# Building blocks and composability

\autoref{fig:packages} depicts the relation among GridapDistributed and other packages in the Julia package ecosystem.

The core library Gridap provides all necessary abstraction and interfaces needed for the FE solution of PDEs (see @Verdugo:2021) for serial computing. GridapDistributed provides distributed-memory counterparts for these abstractions, while leveraging the serial implementations in Gridap to handle the local portion on each parallel task. GridapDistributed relies on PartitionedArrays [@parrays] in order to handle the parallel execution model (e.g., message-passing via the Message Passing Interface (MPI) [@mpi40]), global data distribution layout, and communication among tasks. PartitionedArrays also provides a parallel implementation of partitioned global linear systems (i.e., linear algebra vectors and sparse matrices) as needed in grid-based numerical simulations.
This parallel framework does however not include any performant solver for the resulting linear systems. This was delegated to GridapPETSc, which provides a plethora of highly-scalable and efficient algebraic solvers through a high-level interface to the Portable, Extensible Toolkit for Scientific Computation (PETSc) [@petsc-user-ref].

GridapSolvers complements GridapPETSc with a modular and extensible interface for the design of geometric solvers. Some of the highlights of the library are:

- A set of HPC-first implementations for popular Krylov-based iterative solvers. These solvers extend Gridap's API and are fully compatible with PartitionedArrays.
- A modular, high-level interface for designing block-based preconditioners for multiphysics problems. These preconditioners can be used together with any solver compliant with Gridap's API, including those provided by GridapPETSc.
- A generic interface to handle multi-level distributed meshes, with full support for Adaptative Mesh Refinement (AMR) through GridapP4est. It also provides a modular implementation of geometric multigrid (GMG) solvers, allowing different types of smoothers and restriction/prolongation operators.
- A generic interface for patch-based subdomain decomposition methods, and an implementation of patch-based smoothers for geometric multigrid solvers.

![GridapSolvers and its relation to other packages in the Julia package ecosystem. In this diagram, each node represents  a Julia package, while the (directed) arrows represent relations (dependencies) among packages. Dashed arrows mean the package can be used, but is not necessary. \label{fig:packages}](packages.png){ width=60% }

# Demo

The following code snippet shows how to solve a 2D Stokes cavity problem in a cartesian domain $\Omega = [0,1]^2$. We discretize the velocity and pressure in $H^1(\Omega)$ and $L^2(\Omega)$ respectively, and use the well known stable element pair $Q_k \times P_{k-1}$ with $k=2$. For the cavity problem, we fix the velocity to $u_b = \vec{0}$ and $u_t = \hat{x}$ on the bottom and top boundaries respectively, and homogeneous Neumann boundary conditions elsewhere.
The system is block-assembled and solved using a GMRES solver, right-preconditioned with block-triangular Shur-complement-based preconditioner. The Shur complement is approximated by a mass matrix, and solved using a CG solver with Jacobi preconditioner. The eliminated velocity block is approximated by a 2-level V-cycle Geometric Multigrid solver.
The code is setup to run in parallel with 4 MPI tasks and can be executed with the following command: `mpiexec -n 4 julia --project=. demo.jl`.

```julia
Code in `demo.jl`.
```

# Parallel scaling benchmark

# Acknowledgements

This research was partially funded by the Australian Government through the Australian Research Council (project number DP210103092), the European Commission under the FET-HPC ExaQUte project (Grant agreement ID: 800898) within the Horizon 2020 Framework Program and the project RTI2018-096898-B-I00 from the “FEDER/Ministerio de Ciencia e Innovación (MCIN) – Agencia Estatal de Investigación (AEI)”. F. Verdugo acknowledges support from the “Severo Ochoa Program for Centers of Excellence in R&D (2019-2023)" under the grant CEX2018-000797-S funded by MCIN/AEI/10.13039/501100011033. This work was also supported by computational resources provided by the Australian Government through NCI under the National Computational Merit Allocation Scheme (NCMAS).

# References
