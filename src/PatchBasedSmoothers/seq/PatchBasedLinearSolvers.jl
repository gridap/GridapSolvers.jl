
struct PatchBasedLinearSolver{A,B} <: Gridap.Algebra.LinearSolver
  bilinear_form  :: Function
  Ph             :: A
  Vh             :: B
  M              :: Gridap.Algebra.LinearSolver
end

struct PatchBasedSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: PatchBasedLinearSolver
end

function Gridap.Algebra.symbolic_setup(ls::PatchBasedLinearSolver,mat::AbstractMatrix)
  return PatchBasedSymbolicSetup(ls)
end

struct PatchBasedSmootherNumericalSetup{A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver   :: PatchBasedLinearSolver
  Ap       :: A
  Ap_ns    :: B
  weights  :: C
  caches   :: D
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::AbstractMatrix)
  Ph, Vh = ss.solver.Ph, ss.solver.Vh
  weights = compute_weight_operators(Ph,Vh)

  # Assemble patch system
  assembler = SparseMatrixAssembler(Ph,Ph)
  Ap        = assemble_matrix(ss.solver.bilinear_form,assembler,Ph,Ph)

  # Patch system solver
  Ap_solver = ss.solver.M
  Ap_ss     = symbolic_setup(Ap_solver,Ap)
  Ap_ns     = numerical_setup(Ap_ss,Ap)

  # Caches
  caches = _patch_based_solver_caches(Ph,Vh,Ap)
  
  return PatchBasedSmootherNumericalSetup(ss.solver,Ap,Ap_ns,weights,caches)
end

function _patch_based_solver_caches(Ph::PatchFESpace,Vh::FESpace,Ap::AbstractMatrix)
  rp        = allocate_row_vector(Ap)
  dxp       = allocate_col_vector(Ap)
  return rp, dxp
end

function _patch_based_solver_caches(Ph::GridapDistributed.DistributedSingleFieldFESpace,
                                    Vh::GridapDistributed.DistributedSingleFieldFESpace,
                                    Ap::PSparseMatrix)
  rp      = pfill(0.0,partition(Ph.gids))
  dxp     = pfill(0.0,partition(Ph.gids))
  r       = pfill(0.0,partition(Vh.gids))
  x       = pfill(0.0,partition(Vh.gids))
  return rp, dxp, r, x
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
  consistent!(r) |> fetch
  prolongate!(rp,Ph,r)
  solve!(dxp,Ap_ns,rp)
  inject!(x,Ph,dxp,w,w_sums)
  copy!(x_mat,x)

  return x_mat
end
