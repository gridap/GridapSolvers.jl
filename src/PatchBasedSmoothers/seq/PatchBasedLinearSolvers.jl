
struct PatchBasedLinearSolver{A,B,C} <: Gridap.Algebra.LinearSolver
  biform       :: Function
  patch_space  :: A
  space        :: B
  local_solver :: C
  is_nonlinear :: Bool
  weighted     :: Bool
  
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

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::AbstractMatrix)
  solver = ss.solver
  Ph, Vh = solver.patch_space, solver.space
  weights = solver.weighted ? compute_weight_operators(Ph,Vh) : nothing
  
  ap(u,v) = solver.is_nonlinear ? solver.biform(zero(Vh),u,v) : solver.biform(u,v)

  assem = SparseMatrixAssembler(Ph,Ph)
  Ap    = assemble_matrix(ap,assem,Ph,Ph)
  Ap_ns = numerical_setup(symbolic_setup(solver.local_solver,Ap),Ap)

  # Caches
  rp     = allocate_in_range(Ap)
  dxp    = allocate_in_domain(Ap)
  caches = (rp,dxp)
  
  Ap = solver.is_nonlinear ? Ap : nothing # If linear, we don't need to keep the matrix
  return PatchBasedSmootherNumericalSetup(solver,Ap,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::PSparseMatrix)
  solver = ss.solver
  Ph, Vh = solver.patch_space, solver.space
  weights = solver.weighted ? compute_weight_operators(Ph,Vh) : nothing

  # Patch system solver (only local systems need to be solved)
  ap(u,v) = solver.is_nonlinear ? solver.biform(zero(Vh),u,v) : solver.biform(u,v)
  u, v = get_trial_fe_basis(Vh), get_fe_basis(Vh)
  matdata = collect_cell_matrix(Ph,Ph,ap(u,v))
  Ap, Ap_ns = map(local_views(Ph),matdata) do Ph, matdata
    assem = SparseMatrixAssembler(Ph,Ph)
    Ap    = assemble_matrix(assem,matdata)
    Ap_ns = numerical_setup(symbolic_setup(solver.local_solver,Ap),Ap)
    return Ap, Ap_ns
  end |> PartitionedArrays.tuple_of_arrays

  # Caches
  rp     = pfill(0.0,partition(Ph.gids))
  dxp    = pfill(0.0,partition(Ph.gids))
  r      = pfill(0.0,partition(Vh.gids))
  x      = pfill(0.0,partition(Vh.gids))
  caches = (rp,dxp,r,x)
  
  Ap = solver.is_nonlinear ? Ap : nothing
  return PatchBasedSmootherNumericalSetup(solver,Ap,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup!(ns::PatchBasedSmootherNumericalSetup, A::PSparseMatrix, x::PVector)
  @check ns.solver.is_nonlinear
  solver = ns.solver
  Ph, Vh = solver.patch_space, solver.space
  Ap, Ap_ns = ns.local_A, ns.local_ns

  u0 = FEFunction(Vh,x)
  ap(u,v) = solver.biform(u0,u,v)

  matdata = collect_cell_matrix(Ph,Ph,ap)
  map(Ap, Ap_ns, local_views(Ph), matdata) do Ap, Ap_ns, Ph, matdata
    assem = SparseMatrixAssembler(Ph,Ph)
    assemble_matrix!(Ap,assem,matdata)
    numerical_setup!(Ap_ns,Ap)
  end
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
