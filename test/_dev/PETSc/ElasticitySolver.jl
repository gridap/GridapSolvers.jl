using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.ReferenceFEs
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR
using MPI

import GridapDistributed.DistributedCellField
import GridapDistributed.DistributedFESpace
import GridapDistributed.DistributedDiscreteModel
import GridapDistributed.DistributedMeasure
import GridapDistributed.DistributedTriangulation

function isotropic_3d(E::M,nu::M) where M<:AbstractFloat
    λ = E*nu/((1+nu)*(1-2nu)); μ = E/(2*(1+nu))
    C =[λ+2μ   λ      λ      0      0      0
        λ     λ+2μ    λ      0      0      0
        λ      λ     λ+2μ    0      0      0
        0      0      0      μ      0      0
        0      0      0      0      μ      0
        0      0      0      0      0      μ];
    return SymFourthOrderTensorValue(
        C[1,1], C[6,1], C[5,1], C[2,1], C[4,1], C[3,1],
        C[1,6], C[6,6], C[5,6], C[2,6], C[4,6], C[3,6],
        C[1,5], C[6,5], C[5,5], C[2,5], C[4,5], C[3,5],
        C[1,2], C[6,2], C[5,2], C[2,2], C[4,2], C[3,2],
        C[1,4], C[6,4], C[5,4], C[2,4], C[4,4], C[3,4],
        C[1,3], C[6,3], C[5,3], C[2,3], C[4,3], C[3,3])
end

struct ElasticitySolver{A,B} <: Gridap.Algebra.LinearSolver
  trian ::A
  space ::B
  rtol  ::PetscScalar
  maxits::PetscInt
  function ElasticitySolver(trian::DistributedTriangulation,
                            space::DistributedFESpace;
                            rtol=1.e-12,
                            maxits=100)
    A = typeof(trian)
    B = typeof(space)
    new{A,B}(trian,space,rtol,maxits)
  end
end

struct ElasticitySymbolicSetup{A} <: Gridap.Algebra.SymbolicSetup
  solver::A
end

function Gridap.Algebra.symbolic_setup(solver::ElasticitySolver,A::AbstractMatrix)
  ElasticitySymbolicSetup(solver)
end

function get_dof_coords(trian,space)
  coords = map(local_views(trian),local_views(space),partition(space.gids)) do trian, space, dof_indices
    node_coords = Gridap.Geometry.get_node_coordinates(trian)
    dof_to_node = space.metadata.free_dof_to_node
    dof_to_comp = space.metadata.free_dof_to_comp

    o2l_dofs = own_to_local(dof_indices)
    coords = Vector{PetscScalar}(undef,length(o2l_dofs))
    for (i,dof) in enumerate(o2l_dofs)
      node = dof_to_node[dof]
      comp = dof_to_comp[dof]
      coords[i] = node_coords[node][comp]
    end
    return coords
  end
  ngdofs  = length(space.gids)
  indices = map(local_views(space.gids)) do dof_indices
    owner = part_id(dof_indices)
    own_indices   = OwnIndices(ngdofs,owner,own_to_global(dof_indices))
    ghost_indices = GhostIndices(ngdofs,Int64[],Int32[]) # We only consider owned dofs
    OwnAndGhostIndices(own_indices,ghost_indices)   
  end
  return PVector(coords,indices)
end

function elasticity_ksp_setup(ksp,rtol,maxits)
  rtol = PetscScalar(rtol)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = PetscInt(maxits)

  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
end

function Gridap.Algebra.numerical_setup(ss::ElasticitySymbolicSetup,A::PSparseMatrix)
  s = ss.solver; Dc = num_cell_dims(s.trian)

  # Compute  coordinates for owned dofs
  dof_coords = convert(PETScVector,get_dof_coords(s.trian,s.space))
  @check_error_code GridapPETSc.PETSC.VecSetBlockSize(dof_coords.vec[],Dc)

  # Create matrix nullspace
  B = convert(PETScMatrix,A)
  null = Ref{GridapPETSc.PETSC.MatNullSpace}()
  @check_error_code GridapPETSc.PETSC.MatNullSpaceCreateRigidBody(dof_coords.vec[],null)
  @check_error_code GridapPETSc.PETSC.MatSetNearNullSpace(B.mat[],null[])

  # Setup solver and preconditioner
  ns = GridapPETSc.PETScLinearSolverNS(A,B)
  @check_error_code GridapPETSc.PETSC.KSPCreate(B.comm,ns.ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  elasticity_ksp_setup(ns.ksp,s.rtol,s.maxits)
  @check_error_code GridapPETSc.PETSC.KSPSetUp(ns.ksp[])
  GridapPETSc.Init(ns)
end

############

D = 3
order = 1
n = 20

np_x_dim = 1
np = Tuple(fill(np_x_dim,D)) #Tuple([fill(np_x_dim,D-1)...,1])
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

n_tags = (D==2) ? "tag_6" : "tag_22"
d_tags = (D==2) ? ["tag_5"] : ["tag_21"]

nc = (D==2) ? (n,n) : (n,n,n)
domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
model  = CartesianDiscreteModel(domain,nc)
Ω = Triangulation(model)
Γ = Boundary(model,tags=n_tags)
ΓD = Boundary(model,tags=d_tags)

poly  = (D==2) ? QUAD : HEX
reffe = LagrangianRefFE(VectorValue{D,Float64},poly,order)
V = TestFESpace(model,reffe;dirichlet_tags=d_tags)
U = TrialFESpace(V)
assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},U,V)#,FullyAssembledRows())

dΩ = Measure(Ω,2*order)
dΓ = Measure(Γ,2*order)
C = (D == 2) ? isotropic_2d(1.,0.3) : isotropic_3d(1.,0.3)
g = (D == 2) ? VectorValue(0.0,1.0) : VectorValue(0.0,0.0,1.0)
a(u,v) = ∫((C ⊙ ε(u) ⊙ ε(v)))dΩ
l(v)   = ∫(v ⋅ g)dΓ

op   = AffineFEOperator(a,l,U,V,assem)
A, b = get_matrix(op), get_vector(op);

dim, coords = get_coords(Ω,V);
pcoords = PVector(coords,partition(axes(A,1)))

options = "
  -ksp_type cg -ksp_rtol 1.0e-12
  -pc_type gamg -mat_block_size $D
  -ksp_converged_reason -ksp_error_if_not_converged true
  "
GridapPETSc.with(args=split(options)) do
  solver = PETScLinearSolver(ksp_setup)
  ss = symbolic_setup(solver,A)
  ns = my_numerical_setup(ss,A,pcoords,dim)

  x  = pfill(PetscScalar(1.0),partition(axes(A,2)))
  solve!(x,ns,b)
end
