
struct PatchBasedLinearSolver{A,B,C,D} <: Gridap.Algebra.LinearSolver
  bilinear_form  :: Function
  patch_space    :: A
  space          :: B
  measure        :: C
  local_solver   :: D    
end

struct PatchBasedSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: PatchBasedLinearSolver
end

function Gridap.Algebra.symbolic_setup(ls::PatchBasedLinearSolver,A::AbstractMatrix)
  return PatchBasedSymbolicSetup(ls)
end

struct PatchBasedSmootherNumericalSetup{A,B,C} <: Gridap.Algebra.NumericalSetup
  solver   :: PatchBasedLinearSolver
  local_ns :: A
  weights  :: B
  caches   :: C
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::AbstractMatrix)
  solver = ss.solver
  Ph, Vh, dΩ, solver = solver.patch_space, solver.space, solver.measure
  weights = compute_weight_operators(Ph,Vh)

  assembler = SparseMatrixAssembler(Ph,Ph)
  ap(u,v)   = solver.bilinear_form(u,v,dΩ)
  Ap        = assemble_matrix(ap,assembler,Ph,Ph)
  Ap_ns     = numerical_setup(symbolic_setup(solver.local_solver,Ap),Ap)

  # Caches
  rp        = allocate_in_range(Ap)
  dxp       = allocate_in_domain(Ap)
  caches    = (rp,dxp)
  
  return PatchBasedSmootherNumericalSetup(solver,Ap_ns,weights,caches)
end

function Gridap.Algebra.numerical_setup(ss::PatchBasedSymbolicSetup,A::PSparseMatrix)
  solver = ss.solver
  Ph, Vh, dΩ = solver.patch_space, solver.space, solver.measure
  #weights = compute_weight_operators(Ph,Vh)

  # Patch system solver (only local systems need to be solved)
  Ap_ns = map(local_views(Ph),local_views(dΩ)) do Ph, dΩ
    assembler = SparseMatrixAssembler(Ph,Ph)
    ap(u,v)   = solver.bilinear_form(u,v,dΩ)
    Ap        = assemble_matrix(ap,assembler,Ph,Ph)
    return numerical_setup(symbolic_setup(solver.local_solver,Ap),Ap)
  end

  # Caches
  rp     = pfill(0.0,partition(Ph.gids))
  dxp    = pfill(0.0,partition(Ph.gids))
  r      = pfill(0.0,partition(Vh.gids))
  x      = pfill(0.0,partition(Vh.gids))
  caches = (rp,dxp,r,x)
  
  return PatchBasedSmootherNumericalSetup(solver,Ap_ns,nothing,caches)
end

function Gridap.Algebra.numerical_setup!(ns::PatchBasedSmootherNumericalSetup, A::AbstractMatrix)
  @notimplemented
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::PatchBasedSmootherNumericalSetup,r::AbstractVector)
  Ap_ns, weights, caches = ns.local_ns, ns.weights, ns.caches
  
  Ph = ns.solver.Ph
  w, w_sums = weights
  rp, dxp = caches

  prolongate!(rp,Ph,r)
  solve!(dxp,Ap_ns,rp)
  inject!(x,Ph,dxp)

  return x
end

function Gridap.Algebra.solve!(x_mat::PVector,ns::PatchBasedSmootherNumericalSetup,r_mat::PVector)
  Ap_ns, weights, caches = ns.local_ns, ns.weights, ns.caches
  
  Ph = ns.solver.patch_space
  w, w_sums = weights
  rp, dxp, r, x = caches

  copy!(r,r_mat)
  prolongate!(rp,Ph,r)
  map(solve!,partition(dxp),Ap_ns,partition(rp))
  inject!(x,Ph,dxp)
  copy!(x_mat,x)

  return x_mat
end
