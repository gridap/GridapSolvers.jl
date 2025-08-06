
struct VankaSolver{A,B,C,D} <: Algebra.LinearSolver
  rows::A
  cols::B
  patch_rows::C
  patch_cols::D
end

function VankaSolver(
  patch_rows::AbstractArray{<:AbstractArray{<:Integer}},
  patch_cols::AbstractArray{<:AbstractArray{<:Integer}}
)
  rows = Base.OneTo(maximum(maximum, patch_rows))
  cols = Base.OneTo(maximum(maximum, patch_cols))
  return VankaSolver(rows, cols, patch_rows, patch_cols)
end

function VankaSolver(assem::PatchAssembler)
  rows, cols = get_rows(assem), get_cols(assem)
  patch_rows = collect(Vector{Int32},assem.strategy.patch_rows)
  patch_cols = collect(Vector{Int32},assem.strategy.patch_cols)
  return VankaSolver(rows, cols, patch_rows, patch_cols)
end

function VankaSolver(assem::GridapDistributed.DistributedPatchAssembler)
  rows, cols = map(partition, assem.axes)
  patch_rows, patch_cols = map(local_views(assem)) do assem
    assem.strategy.patch_rows, assem.strategy.patch_cols
  end |> tuple_of_arrays
  return VankaSolver(rows, cols, patch_rows, patch_cols)
end

function VankaSolver(space::FESpace,ptopo;kwargs...)
  assem = PatchAssembler(ptopo,space,space; kwargs...)
  return VankaSolver(assem)
end

function VankaSolver(space::FESpace;kwargs...)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  ptopo = PatchTopology(model)
  return VankaSolver(space,ptopo;kwargs...)
end

struct VankaSS{A} <: Algebra.SymbolicSetup
  solver::A
end

Algebra.symbolic_setup(s::VankaSolver,mat::AbstractMatrix) = VankaSS(s)

struct VankaNS{A,B,C,D,E} <: Algebra.NumericalSetup
  solver::A
  matrix::B
  patch_rows::C
  patch_cols::D
  cache::E
end

function Algebra.numerical_setup(ss::VankaSS,mat::AbstractMatrix)
  solver = ss.solver
  patch_rows = solver.patch_rows
  patch_cols = solver.patch_cols
  cache = (CachedArray(eltype(mat),1), CachedArray(eltype(mat),2))
  return VankaNS(solver, mat, patch_rows, patch_cols, cache)
end

function Algebra.numerical_setup(ss::VankaSS,mat::PSparseMatrix)
  solver = ss.solver

  new_rows = map(SolverInterfaces.split_indices, solver.rows)
  ghosted_mat = SolverInterfaces.fetch_ghost_rows(mat, new_rows)
  new_cols = partition(axes(ghosted_mat,2))

  patch_rows = reindex_patch_ids(solver.patch_rows, solver.rows, new_rows)
  patch_cols = reindex_patch_ids(solver.patch_cols, solver.cols, new_cols)

  caches = map(partition(mat)) do _
    (CachedArray(eltype(mat),1), CachedArray(eltype(mat),2))
  end
  x_c = pzeros(eltype(mat),new_cols)
  b_c = pzeros(eltype(mat),new_rows)
  cache = (x_c,b_c,caches)
  return VankaNS(solver, ghosted_mat, patch_rows, patch_cols, cache)
end

function reindex_patch_ids(patch_ids_old, ids_old, ids_new)
  old_to_new = map(GridapDistributed.find_local_to_local_map,ids_old,ids_new)
  patch_ids_new = map(patch_ids_old,old_to_new) do patch_ids_old, old_to_new
    collect(Vector{Int32},(old_to_new[old] for old in patch_ids_old))
  end
  return patch_ids_new
end

function Algebra.solve!(x::AbstractVector,ns::VankaNS,b::AbstractVector)
  fill!(x, zero(eltype(x)))
  mat, patch_rows, patch_cols, cache = ns.matrix, ns.patch_rows, ns.patch_cols, ns.cache
  return solve_block_jacobi!(x, mat, b, patch_rows, patch_cols, eachindex(patch_rows), cache)
end

function Algebra.solve!(x::PVector,ns::VankaNS,b::PVector)
  mat, patch_rows, patch_cols, caches = ns.matrix, ns.patch_rows, ns.patch_cols, ns.cache
  x_c, b_c, cache = caches

  copy!(b_c,b)
  consistent!(b_c) |> wait
  fill!(x_c, zero(eltype(x)))
  map(partition(x_c), partition(mat), partition(b_c), patch_rows, patch_cols, cache) do x, mat, b, patch_rows, patch_cols, cache
    solve_block_jacobi!(x, mat, b, patch_rows, patch_cols, eachindex(patch_cols), cache)
  end
  assemble!(x_c) |> wait
  copy!(x, x_c)
  consistent!(x) |> wait
  return x
end

function solve_block_jacobi!(
  x,A,b,
  patch_rows, patch_cols,
  patch_ids = eachindex(patch_rows),
  cache = (CachedArray(eltype(x),1), CachedArray(eltype(A),2))
)
  c_xk, c_Ak = cache

  for patch in patch_ids
    rows = patch_rows[patch]
    cols = patch_cols[patch]
    @check isequal(length(rows),length(cols)) "Block is not square"

    n = length(rows)
    setsize!(c_xk,(n,))
    setsize!(c_Ak,(n,n))
    Ak = c_Ak.array
    bk = c_xk.array
    
    copyto!(Ak,view(A,rows,cols))
    copyto!(bk,view(b,rows))
    f = lu!(Ak,NoPivot();check=false)
    @check issuccess(f) "Factorization failed"
    ldiv!(f,bk)
    
    x[cols] .+= bk
  end

  return x
end
