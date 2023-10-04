
struct PatchBasedLinearSolver{A,B} <: Gridap.Algebra.LinearSolver
  bilinear_form  :: Function
  Ph             :: A
  Vh             :: B
  local_solver   :: Gridap.Algebra.LinearSolver
end

struct PatchBasedSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: PatchBasedLinearSolver
end

function Gridap.Algebra.symbolic_setup(ls::PatchBasedLinearSolver,A::AbstractMatrix)
  return PatchBasedSymbolicSetup(ls)
end

struct PatchBasedSmootherNumericalSetup{A,B,C} <: Gridap.Algebra.NumericalSetup
  solver   :: PatchBasedLinearSolver
  Ap_ns    :: A
  weights  :: B
  caches   :: C
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::AbstractMatrix)
  Ph, Vh, solver = ss.solver.Ph, ss.solver.Vh, ss.solver
  weights = compute_weight_operators(Ph,Vh)

  assembler = SparseMatrixAssembler(Ph,Ph)
  Ap        = assemble_matrix(solver.bilinear_form,assembler,Ph,Ph)
  Ap_ns     = numerical_setup(symbolic_setup(solver.local_solver,Ap),Ap)

  # Caches
  rp        = allocate_row_vector(Ap)
  dxp       = allocate_col_vector(Ap)
  caches    = (rp,dxp)
  
  return PatchBasedSmootherNumericalSetup(solver,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::PSparseMatrix)
  Ph, Vh, solver = ss.solver.Ph, ss.solver.Vh, ss.solver
  weights = compute_weight_operators(Ph,Vh)

  # Patch system solver
  # Only local systems need to be solved
  u = get_trial_fe_basis(Ph)
  v = get_fe_basis(Ph)
  matdata = collect_cell_matrix(Ph,Ph,solver.bilinear_form(u,v))
  Ap_ns = map(local_views(Ph),matdata) do Ph, matdata
    assemb = SparseMatrixAssembler(Ph,Ph)
    Ap     = assemble_matrix(assemb,matdata)
    return numerical_setup(symbolic_setup(solver.local_solver,Ap),Ap)
  end

  # Caches
  rp     = pfill(0.0,partition(Ph.gids))
  dxp    = pfill(0.0,partition(Ph.gids))
  r      = pfill(0.0,partition(Vh.gids))
  x      = pfill(0.0,partition(Vh.gids))
  caches = (rp,dxp,r,x)
  
  return PatchBasedSmootherNumericalSetup(solver,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup!(ns::PatchBasedSmootherNumericalSetup, A::AbstractMatrix)
  @notimplemented
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::PatchBasedSmootherNumericalSetup,r::AbstractVector)
  Ap_ns, weights, caches = ns.Ap_ns, ns.weights, ns.caches
  
  Ph = ns.solver.Ph
  w, w_sums = weights
  rp, dxp = caches

  prolongate!(rp,Ph,r)
  solve!(dxp,Ap_ns,rp)
  inject!(x,Ph,dxp,w,w_sums)

  return x
end

function Gridap.Algebra.solve!(x_mat::PVector,ns::PatchBasedSmootherNumericalSetup,r_mat::PVector)
  Ap_ns, weights, caches = ns.Ap_ns, ns.weights, ns.caches
  
  Ph = ns.solver.Ph
  w, w_sums = weights
  rp, dxp, r, x = caches

  copy!(r,r_mat)
  prolongate!(rp,Ph,r)
  map(solve!,partition(dxp),Ap_ns,partition(rp))
  inject!(x,Ph,dxp,w,w_sums)
  copy!(x_mat,x)

  return x_mat
end
