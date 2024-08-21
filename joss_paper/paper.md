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
  - name: Alberto F. Martín
    orcid: 0000-0001-5751-4561
    equal-contrib: true
    affiliation: "2"
  - name: Santiago Badia
    orcid: 0000-0003-2391-4086
    affiliation: "1"
affiliations:
 - name: School of Mathematics, Monash University, Clayton, Victoria, 3800, Australia.
   index: 1
 - name: School of Computing, Australian National University, Canberra, ACT, 2600, Australia
   index: 2
date: 1 August 2024
bibliography: paper.bib

aas-journal: Journal of Open Source Software
---

## Summary and statement of need

The ever-increasing demand for resolution and accuracy in mathematical models of physical processes governed by systems of Partial Differential Equations (PDEs) can only be addressed using fully-parallel advanced numerical discretization methods and scalable solvers, thus able to exploit the vast amount of computational resources in state-of-the-art supercomputers.

One of the biggest scalability bottlenecks within Finite Element (FE) parallel codes is the solution of linear systems arising from the discretization of PDEs.
The implementation of exact factorization-based solvers in parallel environments is an extremely challenging task, and even state-of-the-art libraries such as MUMPS [@MUMPS1; @MUMPS2] or PARDISO [@PARDISO] have severe limitations in terms of scalability and memory consumption above a certain number of CPU cores.
Hence the use of iterative methods is crucial to maintain scalability of FE codes. Unfortunately, the convergence of iterative methods is not guaranteed and rapidly deteriorates as the size of the linear system increases. To retain performance, the use of highly scalable preconditioners is mandatory.
For simple problems, algebraic solvers and preconditioners (i.e based uniquelly on the algebraic system) are enough to obtain robust convergence. Many well-known libraries providing algebraic solvers already exist, such as PETSc [@petsc-user-ref], Trilinos [@trilinos], or Hypre [@hypre]. However, algebraic solvers are not always suited to deal with more challenging problems.

In these cases, solvers that exploit the physics and mathematical discretization of the particular problem are required. This is the case of many multiphysics problems involving differential operators with a large kernel such as the divergence [@Arnold1] and the curl [@Arnold2]. Examples can be found amongst highly relevant problems such as Navier-Stokes, Maxwell or Darcy. Scalable solvers for this type of multiphysics problems rely on exploiting the block structure of such systems to find a spectrally equivalent block-preconditioner, and are often tied to a specific discretization of the underlying equations.

As a consequence, high-quality open-source parallel finite element packages like FEniCS [@fenics-book] or deal.II [@dealII93] already provide implementations of several state-of-the-art physics-informed solvers [@fenics-patch; @dealII-patch]. The Gridap ecosystem [@Badia2020] aims to provide a similar level of functionality within the Julia programming language [@Bezanson2017].

To this end, GridapSolvers is a registered Julia software package which provides highly scalable physics-informed solvers tailored for the FE numerical solution of PDEs on parallel computers within the Gridap ecosystem of packages. Emphasis is put on the modular design of the library, which easily allows new preconditioners to be designed from the user's specific problem.

## Building blocks and composability

\autoref{fig:packages} depicts the relation among GridapDistributed and other packages in the Julia package ecosystem.

The core library Gridap [@Badia2020] provides all necessary abstraction and interfaces needed for the FE solution of PDEs [@Verdugo2021] for serial computing. GridapDistributed [@gridapdistributed] provides distributed-memory counterparts for these abstractions, while leveraging the serial implementations in Gridap to handle the local portion on each parallel task. GridapDistributed relies on PartitionedArrays [@parrays] in order to handle the parallel execution model (e.g., message-passing via the Message Passing Interface (MPI) [@mpi40]), global data distribution layout, and communication among tasks. PartitionedArrays also provides a parallel implementation of partitioned global linear systems (i.e., linear algebra vectors and sparse matrices) as needed in grid-based numerical simulations.
This parallel framework does however not include any performant solver for the resulting linear systems. This was delegated to GridapPETSc, which provides a plethora of highly-scalable and efficient algebraic solvers through a high-level interface to the Portable, Extensible Toolkit for Scientific Computation (PETSc) [@petsc-user-ref].

GridapSolvers complements GridapPETSc with a modular and extensible interface for the design of physics-informed solvers. Some of the highlights of the library are:

- A set of HPC-first implementations for popular Krylov-based iterative solvers. These solvers extend Gridap's API and are fully compatible with PartitionedArrays.
- A modular, high-level interface for designing block-based preconditioners for multiphysics problems. These preconditioners can be used together with any solver compliant with Gridap's API, including those provided by GridapPETSc.
- A generic interface to handle multi-level distributed meshes, with full support for Adaptative Mesh Refinement (AMR) using p4est [@p4est] through GridapP4est.
- A modular implementation of Geometric MultiGrid (GMG) solvers [@gmg-book], allowing different types of smoothers and restriction/prolongation operators.
- A generic interface for patch-based subdomain decomposition methods, and an implementation of patch-based smoothers for GMG solvers.

![GridapSolvers and its relation to other packages in the Julia package ecosystem. In this diagram, each node represents  a Julia package, while the (directed) arrows represent relations (dependencies) among packages. Dashed arrows mean the package can be used, but is not required. \label{fig:packages}](packages.png){ width=50% }

## Demo

The following code snippet shows how to solve a 2D incompressible Stokes cavity problem in a cartesian domain $\Omega = [0,1]^2$. We discretize the velocity and pressure in $H^1(\Omega)$ and $L^2(\Omega)$ respectively, and use the well known stable element pair $Q_k \times P_{k-1}^{-}$ with $k=2$. For the cavity problem, we fix the velocity to $u_t = \hat{x}$ on the top boundary $\Gamma_t = (0,1)\times\{1\}$, and homogeneous Dirichlet boundary conditions elsewhere. We impose a zero-mean pressure constraint to have a solvable system of equations. Given discrete spaces $V \times Q_0$, we find $(u,p) \in V \times Q_0$ such that

$$
  \int_{\Omega} \nabla v : \nabla u - (\nabla \cdot v) p - (\nabla \cdot u) q = \int_{\Omega} v \cdot f \quad \forall v \in V_0, q \in Q_0
$$

where $V_0$ is the space of velocity functions with homogeneous boundary conditions everywhere.

The system is block-assembled and solved using a flexible Generalised Minimum Residual (F-GMRES) solver, together with a block-triangular Shur-complement-based preconditioner. We eliminate the velocity block and approximate the resulting Shur complement by a pressure mass matrix. A more detailed overview of this preconditioner as well as it's spectral analysis can be found in [@Elman2014]. The resulting block structure for the system matrix $\mathcal{A}$ and our preconditioner $\mathcal{P}$ is 

$$
\mathcal{A} = \begin{bmatrix}
  A & B^T \\
  B & 0
\end{bmatrix}
,\quad
\mathcal{P} = \begin{bmatrix}
  A & B^T \\
  0 & -M
\end{bmatrix}
$$

with $A$ the velocity laplacian block, and $M$ a pressure mass matrix.

The mass matrix is approximated by a Conjugate Gradient (CG) solver with Jacobi preconditioner. The eliminated velocity block is approximated by a 2-level V-cycle GMG solver, where the coarsest level is solved exactly in a single processor.
The code for this example can be found [here](https://github.com/gridap/GridapSolvers.jl/tree/joss-paper/joss_paper/demo.jl). It is setup to run in parallel with 4 MPI tasks and can be executed with the following command: `mpiexec -n 4 julia --project=. demo.jl`.

## Parallel scaling benchmark

The following section shows scalability results for the demo problem discussed above. We run our code on the Gadi supercomputer, which is part of the Australian National Computational Infrastructure (NCI). We use Intel's Cascade Lake 2x24-core Xeon Platinum 8274 nodes. Scalability is shown for up to 64 nodes, for a fixed local problem size of 48x64 quadrangle cells per processor. This amounts to a maximum size of approximately 37M cells and 415M degrees of freedom distributed amongst 3072 processors. Within the GMG solver, the number of coarsening levels is progressively increased to keep the global size of the coarsest solve (approximately) constant. The coarsest solve is then performed by a CG solver preconditioned by an Algebraic MultiGrid (AMG) solver, provided by PETSc [@petsc-user-ref] through the package GridapPETSc.jl.

The results in \autoref{fig:scalability} show that the code scales relatively well up to 3072 processors, with loss in performance mostly tied to the number of GMG levels used for the velocity solver. The number of F-GMRES iterations required for convergence is also shown to be relatively constant (and even decreasing for bigger problem sizes), indicating that the preconditioner is robust with respect to the problem size.

The code used to create these results can be found [here](https://github.com/gridap/GridapSolvers.jl/tree/joss-paper/joss_paper/scalability). The exact releases for the packages used are provided by Julia's `Manifest.toml` file.

![**Top**: Weak scalability for a Stokes problem in 2D. Time is given per F-GMRES iteration, as a function of the number of processors. **Middle**: Number of coarsening levels for the GMG solver, as a function of the number of processors. **Bottom**: Number of F-GMRES iterations required for convergence. \label{fig:scalability}](weakScalability.png){ width=80% }

## Acknowledgements

This research was partially funded by the Australian Government through the Australian Research Council (project number DP210103092). This work was also supported by computational resources provided by the Australian Government through NCI under the National Computational Merit Allocation Scheme (NCMAS), the Monash-NCI partnership scheme and the ANU Merit Allocation Scheme (ANUMAS).

## References
