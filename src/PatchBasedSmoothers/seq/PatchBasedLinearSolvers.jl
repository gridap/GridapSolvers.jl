# ON another note. Related to FE assembly. We are going to need:
# "Por otra parte, tb podemos tener metodos q reciben una patch-cell array y la
# aplanan para q parezca una cell array (aunq con cells repetidas). Combinando las
# patch-cell local matrices y cell_dofs aplanadas puedes usar el assembly verbatim si
# quieres ensamblar la matriz."

# Another note. During FE assembly we may end computing the cell matrix of a given cell
# more than once due to cell overlapping among patches (recall the computation of these
# matrices is lazy, it occurs on first touch). Can we live with that or should we pay
# attention on how to avoid this? I think that Gridap already includes tools for
# taking profit of this, I think it is called MemoArray, but it might be something else
# (not 100% sure, to investigate)


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
  caches = _patch_based_solver_caches(Ph,Ap)
  
  return PatchBasedSmootherNumericalSetup(ss.solver,Ap,Ap_ns,weights,caches)
end

function _patch_based_solver_caches(Ph::PatchFESpace,Ap::AbstractMatrix)
  rp        = _allocate_row_vector(Ap)
  dxp       = _allocate_col_vector(Ap)
  return rp, dxp
end

function _patch_based_solver_caches(Ph::GridapDistributed.DistributedSingleFieldFESpace,Ap::PSparseMatrix)
  rp_mat  = _allocate_row_vector(Ap)
  dxp_mat = _allocate_col_vector(Ap)
  rp      = PVector(0.0,Ph.gids)
  dxp     = PVector(0.0,Ph.gids)
  return rp_mat, dxp_mat, rp, dxp
end

function _allocate_col_vector(A::AbstractMatrix)
  zeros(size(A,2))
end

function _allocate_row_vector(A::AbstractMatrix)
  zeros(size(A,1))
end

function _allocate_col_vector(A::PSparseMatrix)
  PVector(0.0,A.cols)
end

function _allocate_row_vector(A::PSparseMatrix)
  PVector(0.0,A.rows)
end

function Gridap.Algebra.numerical_setup!(ns::PatchBasedSmootherNumericalSetup, A::AbstractMatrix)
  Gridap.Helpers.@notimplemented
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::PatchBasedSmootherNumericalSetup,r::AbstractVector)
  Ap_ns, weights, caches = ns.Ap_ns, ns.weights, ns.caches
  
  Ph = ns.solver.Ph
  w, w_sums = weights
  rp, dxp = caches

  prolongate!(rp,Ph,r)
  solve!(dxp,nsAp,rp)
  inject!(x,Ph,dxp,w,w_sums)

  return x
end

function Gridap.Algebra.solve!(x::PVector,ns::PatchBasedSmootherNumericalSetup,r::PVector)
  Ap_ns, weights, caches = ns.Ap_ns, ns.weights, ns.caches
  
  Ph = ns.solver.Ph
  w, w_sums = weights
  rp_mat, dxp_mat, rp, dxp = caches

  prolongate!(rp,Ph,r)
  copy!(rp_mat,rp)
  solve!(dxp_mat,Ap_ns,rp_mat)
  copy!(dxp,dxp_mat)
  inject!(x,Ph,dxp,w,w_sums)

  return x
end
