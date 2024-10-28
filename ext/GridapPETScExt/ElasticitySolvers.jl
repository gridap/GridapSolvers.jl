"""
    struct ElasticitySolver <: LinearSolver
      ...
    end

  GMRES + AMG solver, specifically designed for linear elasticity problems.

  Follows PETSc's documentation for [PCAMG](https://petsc.org/release/manualpages/PC/PCGAMG.html) 
  and [MatNullSpaceCreateRigidBody](https://petsc.org/release/manualpages/Mat/MatNullSpaceCreateRigidBody.html).
"""
struct ElasticitySolver{A} <: Algebra.LinearSolver
  space :: A
  tols  :: SolverTolerances{Float64}

  @doc """
      function ElasticitySolver(space::FESpace; maxiter=500, atol=1.e-12, rtol=1.e-8)

    Returns an instance of [`ElasticitySolver`](@ref) from its underlying properties.
  """
  function ElasticitySolver(space::FESpace;
                            maxiter=500,atol=1.e-12,rtol=1.e-8)
    tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
    A = typeof(space)
    new{A}(space,tols)
  end
end

SolverInterfaces.get_solver_tolerances(s::ElasticitySolver) = s.tols

struct ElasticitySymbolicSetup{A} <: SymbolicSetup
  solver::A
end

function Gridap.Algebra.symbolic_setup(solver::ElasticitySolver,A::AbstractMatrix)
  ElasticitySymbolicSetup(solver)
end

function elasticity_ksp_setup(ksp,tols)
  rtol   = PetscScalar(tols.rtol)
  atol   = PetscScalar(tols.atol)
  dtol   = PetscScalar(tols.dtol)
  maxits = PetscInt(tols.maxiter)

  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)

  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

mutable struct ElasticityNumericalSetup <: NumericalSetup
  A::PETScMatrix
  X::PETScVector
  B::PETScVector
  ksp::Ref{GridapPETSc.PETSC.KSP}
  null::Ref{GridapPETSc.PETSC.MatNullSpace}
  initialized::Bool
  function ElasticityNumericalSetup(A::PETScMatrix,X::PETScVector,B::PETScVector)
    ksp  = Ref{GridapPETSc.PETSC.KSP}()
    null = Ref{GridapPETSc.PETSC.MatNullSpace}()
    new(A,X,B,ksp,null,false)
  end
end

function GridapPETSc.Init(a::ElasticityNumericalSetup)
  @assert Threads.threadid() == 1
  GridapPETSc._NREFS[] += 2
  a.initialized = true
  finalizer(GridapPETSc.Finalize,a)
end

function GridapPETSc.Finalize(ns::ElasticityNumericalSetup)
  if ns.initialized && GridapPETSc.Initialized()
    if ns.A.comm == MPI.COMM_SELF
      @check_error_code GridapPETSc.PETSC.KSPDestroy(ns.ksp)
      @check_error_code GridapPETSc.PETSC.MatNullSpaceDestroy(ns.null)
    else
      @check_error_code GridapPETSc.PETSC.PetscObjectRegisterDestroy(ns.ksp[].ptr)
      @check_error_code GridapPETSc.PETSC.PetscObjectRegisterDestroy(ns.null[].ptr)
    end
    ns.initialized = false
    @assert Threads.threadid() == 1
    GridapPETSc._NREFS[] -= 2
  end
  nothing
end

function Gridap.Algebra.numerical_setup(ss::ElasticitySymbolicSetup,_A::PSparseMatrix)
  _num_dims(space::FESpace) = num_cell_dims(get_triangulation(space))
  _num_dims(space::GridapDistributed.DistributedSingleFieldFESpace) = getany(map(_num_dims,local_views(space)))
  s = ss.solver

  # Create ns 
  A  = convert(PETScMatrix,_A)
  X  = convert(PETScVector,allocate_in_domain(_A))
  B  = convert(PETScVector,allocate_in_domain(_A))
  ns = ElasticityNumericalSetup(A,X,B)

  # Compute  coordinates for owned dofs
  dof_coords = convert(PETScVector,get_dof_coordinates(s.space))
  @check_error_code GridapPETSc.PETSC.VecSetBlockSize(dof_coords.vec[],_num_dims(s.space))

  # Create matrix nullspace
  @check_error_code GridapPETSc.PETSC.MatNullSpaceCreateRigidBody(dof_coords.vec[],ns.null)
  @check_error_code GridapPETSc.PETSC.MatSetNearNullSpace(ns.A.mat[],ns.null[])
  
  # Setup solver and preconditioner
  @check_error_code GridapPETSc.PETSC.KSPCreate(ns.A.comm,ns.ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  elasticity_ksp_setup(ns.ksp,s.tols)
  @check_error_code GridapPETSc.PETSC.KSPSetUp(ns.ksp[])
  GridapPETSc.Init(ns)
end

function Gridap.Algebra.numerical_setup!(ns::ElasticityNumericalSetup,A::AbstractMatrix)
  ns.A = convert(PETScMatrix,A)
  @check_error_code GridapPETSc.PETSC.MatSetNearNullSpace(ns.A.mat[],ns.null[])
  @check_error_code GridapPETSc.PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  @check_error_code GridapPETSc.PETSC.KSPSetUp(ns.ksp[])
  ns
end

function Algebra.solve!(x::AbstractVector{PetscScalar},ns::ElasticityNumericalSetup,b::AbstractVector{PetscScalar})
  X, B = ns.X, ns.B
  copy!(B,b)
  @check_error_code GridapPETSc.PETSC.KSPSolve(ns.ksp[],B.vec[],X.vec[])
  copy!(x,X)
  return x
end