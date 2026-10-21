############################################################################################
# Block-Jacobi patch solver with static condensation
#
# Same local solves as `BlockJacobiSolver` (x = Σ_p R_pᵀ A_pp⁻¹ R_p b),
# with every patch solve computed by first eliminating the patch's interior dofs.
#
# The interior dofs come as global BLOCKS: disjoint sets of dofs that A does not couple to each
# other, typically the dofs interior to a cell or to a group of cells.
# A block is condensed in every patch that contains all of its dofs; the other patch dofs are
# the patch's B dofs. Writing Γ_k for the dofs that A couples to block k,
#
#   W_k = A_kk⁻¹ A_kΓ,   E_k = A_Γk,   S_p = A_BB − Σ_k E_k W_k   (restricted to B),
#
#   y_k = A_kk⁻¹ b_k,   S_p x_B = b_B − Σ_k E_k y_k,   x_k = y_k − W_k x_B,
#
# which is the exact solve of A_pp. The factor of A_kk, W_k and E_k depend only on the block,
# so they are shared by all the patches containing it; S_p is per patch, of the size of B.
# Likewise y_k and E_k y_k are computed once per block in every solve.
#
# The block data k -> (factor of A_kk, W_k, E_k) is a `StaticCondensationArray`, and the patch
# data p -> (block data of p, factor of S_p) a `PatchStaticCondensationArray`. Both compute their
# entries on access, and `caching` picks which of them `numerical_setup` collects:
#   :none      neither; each solve! rebuilds everything patch by patch (memory of one patch).
#   :interior  the block data; S_p is rebuilt and factored per solve.
#   :all       both (one solve! is then only triangular solves and mat-vecs).
#
# The pattern of A must be structurally symmetric (rows = cols, one FE space).

struct CondensedBlockJacobiSolver{A,B,C} <: Algebra.LinearSolver
  rows      :: A
  patch_ids :: B
  block_ids :: C
  caching   :: Symbol
  function CondensedBlockJacobiSolver(
    rows,
    patch_ids::AbstractArray{<:AbstractArray},
    block_ids::AbstractArray{<:AbstractArray};
    caching = :all
  )
    @check caching ∈ (:none, :interior, :all) "Unknown caching $caching"
    patch_table, block_table = ids_to_table(patch_ids), ids_to_table(block_ids)
    A, B, C = typeof(rows), typeof(patch_table), typeof(block_table)
    new{A,B,C}(rows, patch_table, block_table, caching)
  end
end

ids_to_table(ids::AbstractArray{<:AbstractArray{<:Integer}}) = Table(ids)
ids_to_table(ids::AbstractArray{<:AbstractArray{<:AbstractArray{<:Integer}}}) = map(Table, ids)

function CondensedBlockJacobiSolver(
  patch_ids::AbstractArray{<:AbstractArray{<:Integer}},
  block_ids::AbstractArray{<:AbstractArray{<:Integer}};
  kwargs...
)
  rows = Base.OneTo(maximum(p -> maximum(p; init = 0), patch_ids; init = 0))
  return CondensedBlockJacobiSolver(rows, patch_ids, block_ids; kwargs...)
end

function CondensedBlockJacobiSolver(S::BlockJacobiSolver, block_ids; kwargs...)
  @check reduce(&, map(==, S.patch_rows, S.patch_cols); init = true) "Condensation needs square patches on one space"
  return CondensedBlockJacobiSolver(S.rows, S.patch_rows, block_ids; kwargs...)
end

# The blocks are the interior dofs of `space` in each patch of `block_ptopo`, whose patches
# must not overlap. The default `block_ptopo` has one patch per cell.
function CondensedBlockJacobiSolver(
  S::BlockJacobiSolver, space::FESpace, block_ptopo = cell_patch_topology(space);
  kwargs...
)
  block_ids = interior_block_ids(space, block_ptopo)
  return CondensedBlockJacobiSolver(S, block_ids; kwargs...)
end

function CondensedBlockJacobiSolver(
  space::FESpace, ptopo, block_ptopo = cell_patch_topology(space);
  caching = :all, kwargs...
)
  S = BlockJacobiSolver(space, ptopo; kwargs...)
  return CondensedBlockJacobiSolver(S, space, block_ptopo; caching)
end

cell_patch_topology(space::FESpace) = PatchTopology(get_background_model(get_triangulation(space)))

interior_block_ids(space::FESpace, block_ptopo::PatchTopology) =
  get_patch_assembly_ids(space, block_ptopo; assembly = :interior)

function interior_block_ids(space::DistributedFESpace, block_ptopo::DistributedPatchTopology)
  map(local_views(space), local_views(block_ptopo)) do space, block_ptopo
    interior_block_ids(space, block_ptopo)
  end
end

# Symbolic setup: the rows and the patch/block structure, which do not depend on the matrix.
# In parallel, the rows of each part are split into own and ghost, and the ids are local to them.
#   patch_ids      p -> dofs of patch p
#   block_ids      k -> dofs of block k
#   patch_blocks   p -> blocks condensed in patch p
#   patch_B        p -> ids of the B dofs of patch p
struct CondensedBlockJacobiSS{A,B,C,D,E,F} <: Algebra.SymbolicSetup
  solver       :: A
  rows         :: B
  patch_ids    :: C
  block_ids    :: D
  patch_blocks :: E
  patch_B      :: F
end

function Algebra.symbolic_setup(solver::CondensedBlockJacobiSolver, mat::SparseMatrixCSC)
  patch_ids, block_ids = solver.patch_ids, solver.block_ids
  patch_blocks, patch_B = condensation_patches(patch_ids, block_ids, size(mat, 1))
  return CondensedBlockJacobiSS(solver, solver.rows, patch_ids, block_ids, patch_blocks, patch_B)
end

function Algebra.symbolic_setup(solver::CondensedBlockJacobiSolver, mat::PSparseMatrix)
  rows = map(SolverInterfaces.split_indices, solver.rows)
  patch_ids = map(Table, reindex_patch_ids(solver.patch_ids, solver.rows, rows))
  block_ids = map(Table, reindex_patch_ids(solver.block_ids, solver.rows, rows))
  patch_blocks, patch_B = map(patch_ids, block_ids, rows) do patch_ids, block_ids, rows
    condensation_patches(patch_ids, block_ids, local_length(rows))
  end |> tuple_of_arrays
  return CondensedBlockJacobiSS(solver, rows, patch_ids, block_ids, patch_blocks, patch_B)
end

function condensation_patches(patch_ids::Table, block_ids::Table, n)
  dof_to_block = flatten_partition(block_ids, n)
  @check count(!iszero, dof_to_block) == length(block_ids.data) "A dof is in two interior blocks"
  patch_blocks = find_patch_blocks(patch_ids, block_ids, dof_to_block)
  patch_B = find_patch_B(patch_ids, patch_blocks, block_ids, dof_to_block)
  return patch_blocks, patch_B
end

# From the pattern of the matrix:
#   block_gamma    k -> Γ_k, the dofs coupled to block k
#   pb_pos, pb_loc j -> for the j-th (patch, block) pair (a data index of patch_blocks): the
#                       positions in Γ_k of the dofs in the patch, and their local ids in B.
function condensation_gamma(mat::SparseMatrixCSC, block_ids::Table, patch_blocks::Table, patch_B::Table)
  n = size(mat, 1)
  dof_to_block = flatten_partition(block_ids, n)

  # Γ_k: the dofs outside block k that A couples to it
  dof_to_nbors = Table(rowvals(mat), SparseArrays.getcolptr(mat))
  @check blocks_are_uncoupled(dof_to_nbors, block_ids, dof_to_block) "Interior blocks are coupled"
  block_gamma = Arrays.merge_entries(
    dof_to_nbors, block_ids;
    acc = SortedSet{Int32}(), post = dofs -> filter(d -> iszero(dof_to_block[d]), dofs)
  )

  pb_pos, pb_loc = find_patch_gamma(patch_B, patch_blocks, block_gamma, n)
  return block_gamma, pb_pos, pb_loc
end

function blocks_are_uncoupled(dof_to_nbors, block_ids, dof_to_block)
  for k in eachindex(block_ids)
    I = view(block_ids, k)
    for d in I
      nbors = view(dof_to_nbors, d)
      all(r -> dof_to_block[r] ∈ (0, k), nbors) || return false
    end
  end
  return true
end

# Calls f(k) for every block k with all its dofs in the dofs of patch p. `count` and `stamp`
# are work arrays over the blocks; `stamp` must not hold p on entry.
function foreach_complete_block(f, dofs, p, dof_to_block, block_size, count, stamp)
  for d in dofs
    k = dof_to_block[d]
    iszero(k) && continue
    if stamp[k] != p
      stamp[k], count[k] = p, 0
    end
    count[k] += 1
    (count[k] == block_size[k]) && f(k)
  end
end

# Blocks condensed in each patch: those with all their dofs in it
function find_patch_blocks(patch_ids::Table, block_ids::Table, dof_to_block)
  block_size = diff(block_ids.ptrs)
  count, stamp = zeros(Int32, length(block_ids)), zeros(Int32, length(block_ids))

  # Count
  ptrs = zeros(Int32, length(patch_ids) + 1)
  for p in eachindex(patch_ids)
    dofs = view(patch_ids, p)
    foreach_complete_block(dofs, p, dof_to_block, block_size, count, stamp) do k
      ptrs[p+1] += 1
    end
  end
  Arrays.length_to_ptrs!(ptrs)

  # Fill
  fill!(stamp, 0)
  data = Vector{Int32}(undef, ptrs[end] - 1)
  for p in eachindex(patch_ids)
    dofs = view(patch_ids, p)
    foreach_complete_block(dofs, p, dof_to_block, block_size, count, stamp) do k
      data[ptrs[p]] = k
      ptrs[p] += 1
    end
  end
  Arrays.rewind_ptrs!(ptrs)

  return Table(data, ptrs)
end

# B dofs of each patch: its dofs outside its condensed blocks
function find_patch_B(patch_ids::Table, patch_blocks::Table, block_ids::Table, dof_to_block)
  block_size = diff(block_ids.ptrs)

  # Count
  ptrs = zeros(Int32, length(patch_ids) + 1)
  for p in eachindex(patch_ids)
    dofs, blocks = view(patch_ids, p), view(patch_blocks, p)
    nI = sum(k -> block_size[k], blocks; init = 0)
    ptrs[p+1] = length(dofs) - nI
  end
  Arrays.length_to_ptrs!(ptrs)

  # Fill, marking the condensed blocks of p with stamp = p
  stamp = zeros(Int32, length(block_ids))
  data = Vector{Int32}(undef, ptrs[end] - 1)
  for p in eachindex(patch_ids)
    dofs, blocks = view(patch_ids, p), view(patch_blocks, p)
    stamp[blocks] .= p
    for d in dofs
      k = dof_to_block[d]
      if iszero(k) || stamp[k] != p
        data[ptrs[p]] = d
        ptrs[p] += 1
      end
    end
  end
  Arrays.rewind_ptrs!(ptrs)

  return Table(data, ptrs)
end

# For each (patch, block) pair: the positions in Γ_k of the B dofs, and their local ids in B
function find_patch_gamma(patch_B::Table, patch_blocks::Table, block_gamma::Table, n)
  dof_to_lid = zeros(Int32, n)

  # Count
  ptrs = zeros(Int32, length(patch_blocks.data) + 1)
  for p in eachindex(patch_B)
    B, blocks = view(patch_B, p), view(patch_blocks, p)
    find_inverse_index_map!(dof_to_lid, B)
    for (j, k) in zip(datarange(patch_blocks, p), blocks)
      Γ = view(block_gamma, k)
      ptrs[j+1] = count(d -> !iszero(dof_to_lid[d]), Γ)
    end
    dof_to_lid[B] .= 0
  end
  Arrays.length_to_ptrs!(ptrs)

  # Fill
  pos = Vector{Int32}(undef, ptrs[end] - 1)
  loc = Vector{Int32}(undef, ptrs[end] - 1)
  for p in eachindex(patch_B)
    B, blocks = view(patch_B, p), view(patch_blocks, p)
    find_inverse_index_map!(dof_to_lid, B)
    for (j, k) in zip(datarange(patch_blocks, p), blocks)
      Γ = view(block_gamma, k)
      o = ptrs[j]
      for (i, d) in enumerate(Γ)
        l = dof_to_lid[d]
        iszero(l) && continue
        pos[o], loc[o] = i, l
        o += 1
      end
    end
    dof_to_lid[B] .= 0
  end

  return Table(pos, ptrs), Table(loc, ptrs)
end

# Dense copy of A[rows,cols] into M, scattering the nonzeros of each column through `lids`,
# a zero work vector over the dofs (zero again on exit).
function dense_block!(M, A::SparseMatrixCSC, rows, cols, lids)
  fill!(M, zero(eltype(M)))
  find_inverse_index_map!(lids, rows)
  rv, nz = rowvals(A), nonzeros(A)
  for (j, c) in enumerate(cols), q in nzrange(A, c)
    i = lids[rv[q]]
    iszero(i) || (M[i, j] = nz[q])
  end
  lids[rows] .= 0
  return M
end

function lu_factor!(M, msg)
  F = lu!(M, NoPivot(); check = false)
  @check issuccess(F) msg
  return F
end

const CondensationLU{T} = LU{T,Matrix{T},Vector{LinearAlgebra.BlasInt}}

# k -> (factor of A_kk, W_k, E_k), computed on access. `lids` is a zero work vector over the dofs.
struct StaticCondensationArray{T,A,B,C} <: AbstractVector{Tuple{CondensationLU{T},Matrix{T},Matrix{T}}}
  matrix      :: A
  block_ids   :: B
  block_gamma :: C
  lids        :: Vector{Int32}
  function StaticCondensationArray(matrix::SparseMatrixCSC{T}, block_ids, block_gamma, lids) where T
    A, B, C = typeof(matrix), typeof(block_ids), typeof(block_gamma)
    new{T,A,B,C}(matrix, block_ids, block_gamma, lids)
  end
end

Base.size(a::StaticCondensationArray) = (length(a.block_ids),)
Base.IndexStyle(::Type{<:StaticCondensationArray}) = IndexLinear()
Base.getindex(a::StaticCondensationArray, k::Integer) = getindex!(array_cache(a), a, k)

Arrays.array_cache(a::StaticCondensationArray{T}) where T = (CachedMatrix(T), CachedMatrix(T), CachedMatrix(T))

function Arrays.getindex!(cache, a::StaticCondensationArray, k::Integer)
  c_F, c_W, c_E = cache
  I, Γ = view(a.block_ids, k), view(a.block_gamma, k)
  setsize!(c_F, (length(I), length(I)))
  setsize!(c_W, (length(I), length(Γ)))
  setsize!(c_E, (length(Γ), length(I)))
  F = lu_factor!(dense_block!(c_F.array, a.matrix, I, I, a.lids), "Interior block factorization failed")
  W = ldiv!(F, dense_block!(c_W.array, a.matrix, I, Γ, a.lids))
  E = dense_block!(c_E.array, a.matrix, Γ, I, a.lids)
  return F, W, E
end

function Arrays.lazy_collect(a::StaticCondensationArray)
  cache = array_cache(a)
  r = Vector{eltype(a)}(undef, length(a))
  for k in eachindex(a)
    F, W, E = getindex!(cache, a, k)
    r[k] = (copy(F), copy(W), copy(E))
  end
  return r
end

# p -> (block data of the blocks condensed in p, factor of S_p), computed on access.
# `block_data` is a StaticCondensationArray or its collection; `patch_S` is nothing (S_p computed
# on access) or the factor of every S_p.
struct PatchStaticCondensationArray{T,A,B,C,D,E} <: AbstractVector{T}
  matrix       :: A
  block_data   :: B
  patch_blocks :: C
  patch_B      :: C
  pb_pos       :: C
  pb_loc       :: C
  lids         :: Vector{Int32}
  patch_S      :: D
  function PatchStaticCondensationArray(
    matrix::SparseMatrixCSC{Tv}, block_data, patch_blocks, patch_B, pb_pos, pb_loc, lids, patch_S = nothing
  ) where Tv
    bdata = view(Vector{eltype(block_data)}(undef, 0), 1:0)
    T = Tuple{typeof(bdata),CondensationLU{Tv}}
    A, B, C, D = typeof(matrix), typeof(block_data), typeof(patch_blocks), typeof(patch_S)
    new{T,A,B,C,D,Tv}(matrix, block_data, patch_blocks, patch_B, pb_pos, pb_loc, lids, patch_S)
  end
end

Base.size(a::PatchStaticCondensationArray) = (length(a.patch_B),)
Base.IndexStyle(::Type{<:PatchStaticCondensationArray}) = IndexLinear()
Base.getindex(a::PatchStaticCondensationArray, p::Integer) = getindex!(array_cache(a), a, p)

# One block cache per block slot: the block data of a patch must be alive at the same time.
function Arrays.array_cache(a::PatchStaticCondensationArray{T,A,B,C,D,Tv}) where {T,A,B,C,D,Tv}
  nmax = maximum(diff(a.patch_blocks.ptrs); init = 0)
  block_caches = [array_cache(a.block_data) for _ in 1:nmax]
  bdata = Vector{eltype(a.block_data)}(undef, nmax)
  schur_caches = (CachedMatrix(Tv), CachedMatrix(Tv), CachedMatrix(Tv), CachedMatrix(Tv))
  return block_caches, bdata, schur_caches
end

function Arrays.getindex!(cache, a::PatchStaticCondensationArray, p::Integer)
  block_caches, bdata, schur_caches = cache
  blocks = view(a.patch_blocks, p)
  for (i, k) in enumerate(blocks)
    bdata[i] = getindex!(block_caches[i], a.block_data, k)
  end
  pdata = view(bdata, 1:length(blocks))
  F_S = isnothing(a.patch_S) ? patch_schur!(schur_caches, a, p, pdata) : a.patch_S[p]
  return pdata, F_S
end

# S_p = A_BB − Σ_k E_k W_k, each term restricted to the B dofs block k couples to.
# `pdata[j]` is the block data of the j-th block of patch p.
function patch_schur!(cache, a::PatchStaticCondensationArray, p, pdata)
  c_S, c_E, c_W, c_EW = cache
  B = view(a.patch_B, p)
  setsize!(c_S, (length(B), length(B)))
  S = dense_block!(c_S.array, a.matrix, B, B, a.lids)
  for (j, (_, W, E)) in zip(datarange(a.patch_blocks, p), pdata)
    pos, loc = view(a.pb_pos, j), view(a.pb_loc, j)
    isempty(pos) && continue
    setsize!(c_E, (length(pos), size(E, 2))); Ep = c_E.array
    setsize!(c_W, (size(W, 1), length(pos))); Wp = c_W.array
    setsize!(c_EW, (length(pos), length(pos)))
    @views Ep .= E[pos, :]
    @views Wp .= W[:, pos]
    EW = mul!(c_EW.array, Ep, Wp)
    @views S[loc, loc] .-= EW
  end
  return lu_factor!(S, "Condensed patch factorization failed")
end

function Arrays.lazy_collect(a::PatchStaticCondensationArray{T,A,B,C,D,Tv}) where {T,A,B,C,D,Tv}
  isnothing(a.patch_S) || return a
  cache = array_cache(a)
  patch_S = Vector{CondensationLU{Tv}}(undef, length(a))
  for p in eachindex(a)
    _, F_S = getindex!(cache, a, p)
    patch_S[p] = copy(F_S)
  end
  return PatchStaticCondensationArray(
    a.matrix, a.block_data, a.patch_blocks, a.patch_B, a.pb_pos, a.pb_loc, a.lids, patch_S
  )
end

mutable struct CondensedBlockJacobiNS{A,B,C,D,E,F,G} <: Algebra.NumericalSetup
  ss          :: A
  matrix      :: B
  block_gamma :: C
  pb_pos      :: D
  pb_loc      :: D
  block_data  :: E   # k -> (factor of A_kk, W_k, E_k)
  patch_data  :: F   # p -> (block data of p, factor of S_p)
  caches      :: G
end

# What `caching` stores: :interior collects the block data, :all also the patch data.
function condensation_data(
  mat::SparseMatrixCSC, caching, block_ids, block_gamma, patch_blocks, patch_B, pb_pos, pb_loc, lids
)
  block_data = StaticCondensationArray(mat, block_ids, block_gamma, lids)
  (caching != :none) && (block_data = Arrays.lazy_collect(block_data))
  patch_data = PatchStaticCondensationArray(mat, block_data, patch_blocks, patch_B, pb_pos, pb_loc, lids)
  (caching == :all) && (patch_data = Arrays.lazy_collect(patch_data))
  return block_data, patch_data
end

# (y, Ey, lids, work vectors): y_k and E_k y_k of every block, flat as block_ids and
# block_gamma; the dof work vector of `dense_block!`; and the per-patch work vectors.
function condensation_caches(mat::SparseMatrixCSC, block_ids, block_gamma)
  T = eltype(mat)
  y  = zeros(T, length(block_ids.data))
  Ey = zeros(T, length(block_gamma.data))
  lids = zeros(Int32, size(mat, 1))
  work = (CachedArray(T, 1), CachedArray(T, 1), CachedArray(T, 1))
  return y, Ey, lids, work
end

function condensation_setup(mat::SparseMatrixCSC, caching, block_ids, patch_blocks, patch_B)
  block_gamma, pb_pos, pb_loc = condensation_gamma(mat, block_ids, patch_blocks, patch_B)
  caches = condensation_caches(mat, block_ids, block_gamma)
  block_data, patch_data = condensation_data(
    mat, caching, block_ids, block_gamma, patch_blocks, patch_B, pb_pos, pb_loc, caches[3]
  )
  return block_gamma, pb_pos, pb_loc, block_data, patch_data, caches
end

function Algebra.numerical_setup(ss::CondensedBlockJacobiSS, mat::SparseMatrixCSC)
  data = condensation_setup(mat, ss.solver.caching, ss.block_ids, ss.patch_blocks, ss.patch_B)
  return CondensedBlockJacobiNS(ss, mat, data...)
end

# The local matrix over the own and ghost rows: the columns of the same dofs, in row order
function local_square_matrices(mat::PSparseMatrix)
  map(partition(mat), partition(axes(mat, 1)), partition(axes(mat, 2))) do A, rows, cols
    row_to_col = GridapDistributed.find_local_to_local_map(rows, cols)
    @check all(c -> c > 0, row_to_col) "A local row has no matching column"
    A[:, row_to_col]
  end
end

function Algebra.numerical_setup(ss::CondensedBlockJacobiSS, mat::PSparseMatrix)
  caching = ss.solver.caching
  new_mat = SolverInterfaces.fetch_ghost_rows(mat, ss.rows)
  block_gamma, pb_pos, pb_loc, block_data, patch_data, caches = map(
    local_square_matrices(new_mat), ss.block_ids, ss.patch_blocks, ss.patch_B
  ) do mat, block_ids, patch_blocks, patch_B
    condensation_setup(mat, caching, block_ids, patch_blocks, patch_B)
  end |> tuple_of_arrays
  x_c = pzeros(eltype(mat), ss.rows)
  b_c = pzeros(eltype(mat), ss.rows)
  return CondensedBlockJacobiNS(
    ss, new_mat, block_gamma, pb_pos, pb_loc, block_data, patch_data, (x_c, b_c, caches)
  )
end

function Algebra.numerical_setup!(ns::CondensedBlockJacobiNS, mat::AbstractMatrix, vec::AbstractVector)
  numerical_setup!(ns, mat)
end

function Algebra.numerical_setup!(ns::CondensedBlockJacobiNS, mat::SparseMatrixCSC)
  ss = ns.ss
  ns.matrix = mat
  ns.block_data, ns.patch_data = condensation_data(
    mat, ss.solver.caching, ss.block_ids, ns.block_gamma, ss.patch_blocks, ss.patch_B, ns.pb_pos, ns.pb_loc, ns.caches[3]
  )
  return ns
end

function Algebra.numerical_setup!(ns::CondensedBlockJacobiNS, mat::PSparseMatrix)
  ss = ns.ss
  caching = ss.solver.caching
  _, _, caches = ns.caches
  ns.matrix = SolverInterfaces.fetch_ghost_rows(mat, ss.rows)
  ns.block_data, ns.patch_data = map(
    local_square_matrices(ns.matrix), ss.block_ids, ns.block_gamma, ss.patch_blocks, ss.patch_B, ns.pb_pos, ns.pb_loc, caches
  ) do mat, block_ids, block_gamma, patch_blocks, patch_B, pb_pos, pb_loc, caches
    condensation_data(mat, caching, block_ids, block_gamma, patch_blocks, patch_B, pb_pos, pb_loc, caches[3])
  end |> tuple_of_arrays
  return ns
end

# y_k = A_kk⁻¹ b_k, Ey_k = E_k y_k
function block_solve!(y, Ey, block_ids, block_gamma, k, F, E, b)
  I = view(block_ids, k)
  yk  = view(y, datarange(block_ids, k))
  Eyk = view(Ey, datarange(block_gamma, k))
  copyto!(yk, view(b, I))
  ldiv!(F, yk)
  mul!(Eyk, E, yk)
end

function Algebra.solve!(x::AbstractVector, ns::CondensedBlockJacobiNS, b::AbstractVector)
  ss = ns.ss
  fill!(x, zero(eltype(x)))
  return solve_condensed_block_jacobi!(
    x, b, ss.block_ids, ns.block_gamma, ss.patch_blocks, ss.patch_B, ns.pb_pos, ns.pb_loc,
    ns.block_data, ns.patch_data, ns.caches
  )
end

function Algebra.solve!(x::PVector, ns::CondensedBlockJacobiNS, b::PVector)
  ss = ns.ss
  x_c, b_c, caches = ns.caches

  copy!(b_c, b)
  consistent!(b_c) |> wait
  fill!(x_c, zero(eltype(x)))
  map(
    solve_condensed_block_jacobi!,
    partition(x_c), partition(b_c), ss.block_ids, ns.block_gamma, ss.patch_blocks, ss.patch_B, ns.pb_pos, ns.pb_loc,
    ns.block_data, ns.patch_data, caches
  )
  assemble!(x_c) |> wait
  copy!(x, x_c)
  consistent!(x) |> wait
  return x
end

function solve_condensed_block_jacobi!(
  x, b, block_ids, block_gamma, patch_blocks, patch_B, pb_pos, pb_loc, block_data, patch_data, caches
)
  y, Ey, _, (c_xB, c_xI, c_xΓ) = caches

  # Block solves, shared by all patches
  block_cache = array_cache(block_data)
  for k in eachindex(block_data)
    F, _, E = getindex!(block_cache, block_data, k)
    block_solve!(y, Ey, block_ids, block_gamma, k, F, E, b)
  end

  cache = array_cache(patch_data)
  for p in eachindex(patch_B)
    B, blocks = view(patch_B, p), view(patch_blocks, p)
    bdata, F_S = getindex!(cache, patch_data, p)

    # Condensed rhs: b_B − Σ_k E_k y_k
    setsize!(c_xB, (length(B),)); xB = c_xB.array
    copyto!(xB, view(b, B))
    for (j, k) in zip(datarange(patch_blocks, p), blocks)
      pos, loc = view(pb_pos, j), view(pb_loc, j)
      Eyk = view(Ey, datarange(block_gamma, k))
      @views xB[loc] .-= Eyk[pos]
    end

    # Interface: x_B = S_p⁻¹ (b_B − Σ_k E_k y_k)
    ldiv!(F_S, xB)
    @views x[B] .+= xB

    # Interiors: x_k = y_k − W_k x_B, with x_B extended by zero to Γ_k
    for (j, k, (_, W, _)) in zip(datarange(patch_blocks, p), blocks, bdata)
      I = view(block_ids, k)
      yk = view(y, datarange(block_ids, k))
      pos, loc = view(pb_pos, j), view(pb_loc, j)
      setsize!(c_xI, (length(I),)); xI = c_xI.array
      copyto!(xI, yk)
      if !isempty(pos)
        setsize!(c_xΓ, (size(W, 2),)); xΓ = c_xΓ.array
        fill!(xΓ, zero(eltype(xΓ)))
        @views xΓ[pos] .= xB[loc]
        mul!(xI, W, xΓ, -1, 1)
      end
      @views x[I] .+= xI
    end
  end
  return x
end

# Number of Float64 values held by the numerical setup (block factors, W_k, E_k, S_p factors).
condensation_storage(ns::CondensedBlockJacobiNS) = condensation_storage(ns.block_data, ns.patch_data)

function condensation_storage(ns::CondensedBlockJacobiNS{A,<:PSparseMatrix}) where A
  reduce(+, map(condensation_storage, ns.block_data, ns.patch_data); init = 0)
end

function condensation_storage(block_data, patch_data)
  n = 0
  if !isa(block_data, StaticCondensationArray)
    n += sum(((F, W, E),) -> length(F.factors) + length(W) + length(E), block_data; init = 0)
  end
  patch_S = patch_data.patch_S
  if !isnothing(patch_S)
    n += sum(F -> length(F.factors), patch_S; init = 0)
  end
  return n
end
