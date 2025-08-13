
struct BlockJacobiSolver{A,B,C,D} <: Algebra.LinearSolver
  rows::A
  cols::B
  patch_rows::C
  patch_cols::D
end

function BlockJacobiSolver(
  patch_rows::AbstractArray{<:AbstractArray{<:Integer}},
  patch_cols::AbstractArray{<:AbstractArray{<:Integer}}
)
  rows = Base.OneTo(maximum(maximum, patch_rows))
  cols = Base.OneTo(maximum(maximum, patch_cols))
  return BlockJacobiSolver(rows, cols, patch_rows, patch_cols)
end

function BlockJacobiSolver(assem::PatchAssembler)
  rows, cols = get_rows(assem), get_cols(assem)
  patch_rows = collect(Vector{Int32},assem.strategy.patch_rows)
  patch_cols = collect(Vector{Int32},assem.strategy.patch_cols)
  return BlockJacobiSolver(rows, cols, patch_rows, patch_cols)
end

function BlockJacobiSolver(assem::GridapDistributed.DistributedPatchAssembler)
  rows, cols = map(partition, assem.axes)
  patch_rows, patch_cols = map(local_views(assem)) do assem
    assem.strategy.patch_rows, assem.strategy.patch_cols
  end |> tuple_of_arrays
  return BlockJacobiSolver(rows, cols, patch_rows, patch_cols)
end

function BlockJacobiSolver(space::FESpace,ptopo;kwargs...)
  assem = PatchAssembler(ptopo,space,space; kwargs...)
  return BlockJacobiSolver(assem)
end

function BlockJacobiSolver(space::FESpace;kwargs...)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  ptopo = PatchTopology(model)
  return BlockJacobiSolver(space,ptopo;kwargs...)
end

struct BlockJacobiSS{A} <: Algebra.SymbolicSetup
  solver::A
end

Algebra.symbolic_setup(s::BlockJacobiSolver,mat::AbstractMatrix) = BlockJacobiSS(s)

mutable struct BlockJacobiNS{A,B,C,D,E} <: Algebra.NumericalSetup
  solver::A
  matrix::B
  patch_rows::C
  patch_cols::D
  cache::E
end

function Algebra.numerical_setup(ss::BlockJacobiSS,mat::AbstractMatrix)
  solver = ss.solver
  patch_rows = solver.patch_rows
  patch_cols = solver.patch_cols
  cache = (CachedArray(eltype(mat),1), CachedArray(eltype(mat),2))
  return BlockJacobiNS(solver, mat, patch_rows, patch_cols, cache)
end

function Algebra.numerical_setup(ss::BlockJacobiSS,mat::PSparseMatrix)
  solver = ss.solver

  new_rows = map(SolverInterfaces.split_indices, solver.rows)
  new_mat  = SolverInterfaces.fetch_ghost_rows(mat, new_rows)
  new_cols = partition(axes(new_mat,2))

  patch_rows = reindex_patch_ids(solver.patch_rows, solver.rows, new_rows)
  patch_cols = reindex_patch_ids(solver.patch_cols, solver.cols, new_cols)

  caches = map(partition(mat)) do _
    (CachedArray(eltype(mat),1), CachedArray(eltype(mat),2))
  end
  x_c = pzeros(eltype(mat),new_cols)
  b_c = pzeros(eltype(mat),new_rows)
  cache = (x_c,b_c,caches)
  return BlockJacobiNS(solver, new_mat, patch_rows, patch_cols, cache)
end

function Algebra.numerical_setup!(ns::BlockJacobiNS,mat::AbstractMatrix)
  ns.matrix = mat
  return ns
end

function Algebra.numerical_setup!(ns::BlockJacobiNS,mat::PSparseMatrix)
  x_c, b_c, caches = ns.cache

  new_rows = partition(axes(b_c,1))
  ns.matrix = SolverInterfaces.fetch_ghost_rows(mat, new_rows)
  new_cols = partition(axes(ns.matrix,2))
  
  same_new_cols = PartitionedArrays.matching_local_indices(PRange(new_cols),axes(x_c,1))
  if !same_new_cols
    ns.patch_cols = reindex_patch_ids(ns.solver.patch_cols, ns.solver.cols, new_cols)
    ns.cache = (pzeros(eltype(mat),new_cols),b_c,caches)
  end

  return ns
end

function reindex_patch_ids(patch_ids_old, ids_old, ids_new)
  old_to_new = map(GridapDistributed.find_local_to_local_map,ids_old,ids_new)
  patch_ids_new = map(patch_ids_old,old_to_new) do patch_ids_old, old_to_new
    collect(Vector{Int32},(old_to_new[old] for old in patch_ids_old))
  end
  return patch_ids_new
end

function Algebra.solve!(x::AbstractVector,ns::BlockJacobiNS,b::AbstractVector)
  fill!(x, zero(eltype(x)))
  mat, patch_rows, patch_cols, cache = ns.matrix, ns.patch_rows, ns.patch_cols, ns.cache
  return solve_block_jacobi!(x, mat, b, patch_rows, patch_cols, eachindex(patch_rows), cache)
end

function Algebra.solve!(x::PVector,ns::BlockJacobiNS,b::PVector)
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
