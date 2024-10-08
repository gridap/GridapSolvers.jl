"""
    struct PatchBasedLinearSolver <: LinearSolver
      ...
    end

Sub-assembled linear solver for patch-based methods. Given a bilinear form `a` and
a space decomposition `V = Σ_i V_i` given by a patch space, returns a global correction
given by aggregated local corrections, i.e 

```
dx = Σ_i w_i I_i inv(A_i) (I_i)^* x 
```

where `A_i` is the patch-local system matrix defined by 

```
(A_i u_i, v_i) = a(u_i,v_i) ∀ v_i ∈ V_i
```

and `I_i` is the natural injection from the patch space
to the global space. The aggregation can be un-weighted (i.e. `w_i = 1`) or weighted, where
`w_i = 1/#(i)`.

"""
struct PatchBasedLinearSolver{A,B,C} <: Gridap.Algebra.LinearSolver
  biform       :: Function
  patch_space  :: A
  space        :: B
  local_solver :: C
  is_nonlinear :: Bool
  weighted     :: Bool
  
  @doc """
      function PatchBasedLinearSolver(
        biform::Function, 
        patch_space::FESpace, 
        space::FESpace;
        local_solver = LUSolver(),
        is_nonlinear = false,
        weighted = false
      )
    
    Returns an instance of [`PatchBasedLinearSolver`](@ref) from its underlying properties.
    Local patch-systems are solved with `local_solver`. If `weighted`, uses weighted 
    patch aggregation to compute the global correction.
  """
  function PatchBasedLinearSolver(
    biform::Function, patch_space::FESpace, space::FESpace;
    local_solver = LUSolver(),
    is_nonlinear = false,
    weighted = false
  )
    A = typeof(patch_space)
    B = typeof(space)
    C = typeof(local_solver)
    return new{A,B,C}(biform,patch_space,space,local_solver,is_nonlinear,weighted)
  end
end

struct PatchBasedSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: PatchBasedLinearSolver
end

function Gridap.Algebra.symbolic_setup(ls::PatchBasedLinearSolver,A::AbstractMatrix)
  return PatchBasedSymbolicSetup(ls)
end

struct PatchBasedSmootherNumericalSetup{A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver   :: PatchBasedLinearSolver
  local_A  :: A
  local_ns :: B
  weights  :: C
  caches   :: D
end

function assemble_patch_matrices(Ph::FESpace,ap;local_solver=LUSolver())
  assem = SparseMatrixAssembler(Ph,Ph)
  Ap    = assemble_matrix(ap,assem,Ph,Ph)
  Ap_ns = numerical_setup(symbolic_setup(local_solver,Ap),Ap)
  return Ap, Ap_ns
end

function assemble_patch_matrices(Ph::GridapDistributed.DistributedFESpace,ap;local_solver=LUSolver())
  u, v  = get_trial_fe_basis(Ph), get_fe_basis(Ph)
  matdata = collect_cell_matrix(Ph,Ph,ap(u,v))
  Ap, Ap_ns = map(local_views(Ph),matdata) do Ph, matdata
    assem = SparseMatrixAssembler(Ph,Ph)
    Ap    = assemble_matrix(assem,matdata)
    Ap_ns = numerical_setup(symbolic_setup(local_solver,Ap),Ap)
    return Ap, Ap_ns
  end |> PartitionedArrays.tuple_of_arrays
  return Ap, Ap_ns
end

function update_patch_matrices!(Ap,Ap_ns,Ph::FESpace,ap)
  assem = SparseMatrixAssembler(Ph,Ph)
  assemble_matrix!(Ap,assem,Ph,Ph,ap)
  numerical_setup!(Ap_ns,Ap)
end

function update_patch_matrices!(Ap,Ap_ns,Ph::GridapDistributed.DistributedFESpace,ap)
  u, v  = get_trial_fe_basis(Ph), get_fe_basis(Ph)
  matdata = collect_cell_matrix(Ph,Ph,ap(u,v))
  map(Ap, Ap_ns, local_views(Ph), matdata) do Ap, Ap_ns, Ph, matdata
    assem = SparseMatrixAssembler(Ph,Ph)
    assemble_matrix!(Ap,assem,matdata)
    numerical_setup!(Ap_ns,Ap)
  end
end

function allocate_patch_workvectors(Ph::FESpace,Vh::FESpace)
  rp     = zero_free_values(Ph)
  dxp    = zero_free_values(Ph)
  return rp,dxp
end

function allocate_patch_workvectors(Ph::GridapDistributed.DistributedFESpace,Vh::GridapDistributed.DistributedFESpace)
  rp     = zero_free_values(Ph)
  dxp    = zero_free_values(Ph)
  r      = zero_free_values(Vh)
  x      = zero_free_values(Vh)
  return rp,dxp,r,x
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::AbstractMatrix)
  @check !ss.solver.is_nonlinear
  solver = ss.solver
  local_solver = solver.local_solver
  Ph, Vh = solver.patch_space, solver.space
  
  ap(u,v) = solver.biform(u,v)
  Ap, Ap_ns = assemble_patch_matrices(Ph,ap;local_solver)
  weights = solver.weighted ? compute_weight_operators(Ph,Vh) : nothing
  caches = allocate_patch_workvectors(Ph,Vh)
  return PatchBasedSmootherNumericalSetup(solver,nothing,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::AbstractMatrix,x::AbstractVector)
  @check ss.solver.is_nonlinear
  solver = ss.solver
  local_solver = solver.local_solver
  Ph, Vh = solver.patch_space, solver.space
  
  u0 = FEFunction(Vh,x)
  ap(u,v) = solver.biform(u0,u,v)
  Ap, Ap_ns = assemble_patch_matrices(Ph,ap;local_solver)
  weights = solver.weighted ? compute_weight_operators(Ph,Vh) : nothing
  caches = allocate_patch_workvectors(Ph,Vh)
  return PatchBasedSmootherNumericalSetup(solver,Ap,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup!(ns::PatchBasedSmootherNumericalSetup,A::AbstractMatrix,x::AbstractVector)
  @check ns.solver.is_nonlinear
  solver = ns.solver
  Ph, Vh = solver.patch_space, solver.space
  Ap, Ap_ns = ns.local_A, ns.local_ns

  u0 = FEFunction(Vh,x)
  ap(u,v) = solver.biform(u0,u,v)
  update_patch_matrices!(Ap,Ap_ns,Ph,ap)
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::PatchBasedSmootherNumericalSetup,r::AbstractVector)
  Ap_ns, weights, caches = ns.local_ns, ns.weights, ns.caches
  
  Ph = ns.solver.patch_space
  #w, w_sums = weights
  rp, dxp = caches

  prolongate!(rp,Ph,r)
  solve!(dxp,Ap_ns,rp)
  inject!(x,Ph,dxp)

  return x
end

function Gridap.Algebra.solve!(x_mat::PVector,ns::PatchBasedSmootherNumericalSetup,r_mat::PVector)
  Ap_ns, weights, caches = ns.local_ns, ns.weights, ns.caches
  
  Ph = ns.solver.patch_space
  #w, w_sums = weights
  rp, dxp, r, x = caches

  copy!(r,r_mat)
  prolongate!(rp,Ph,r)
  map(solve!,partition(dxp),Ap_ns,partition(rp))
  inject!(x,Ph,dxp)
  copy!(x_mat,x)

  return x_mat
end
