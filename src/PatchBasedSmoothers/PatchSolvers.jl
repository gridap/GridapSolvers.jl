
struct PatchSolverFromMats{A,B,C} <: Algebra.LinearSolver
  axes :: A
  patch_axes :: B
  patch_mats :: C
  collect_factorizations :: Bool
end

struct PatchSolverFromWeakform <: Algebra.LinearSolver
  assem :: Assembler
  trial :: FESpace
  test  :: FESpace
  weakform :: Function
  is_nonlinear :: Bool
  collect_factorizations :: Bool
end

function PatchSolver(
  ptopo::Union{PatchTopology,DistributedPatchTopology}, trial::FESpace, test::FESpace, args...; 
  is_nonlinear = false, collect_factorizations = false, assem_kwargs...
)
  assem = PatchAssembler(ptopo, trial, test; assem_kwargs...)
  return PatchSolver(assem, trial, test, args...; is_nonlinear, collect_factorizations)
end

function PatchSolver(
  assem::PatchAssembler, trial::FESpace, test::FESpace, weakform::Function; 
  is_nonlinear = false, collect_factorizations = false, assem_kwargs...
)
  if !is_nonlinear
    axes = (get_free_dof_ids(test), get_free_dof_ids(trial))
    patch_axes = (assem.strategy.patch_rows, assem.strategy.patch_cols)
    patch_mats = assemble_matrix(weakform, assem, trial, test)
    return PatchSolverFromMats(axes, patch_axes, patch_mats, collect_factorizations)
  else
    return PatchSolverFromWeakform(assem, trial, test, weakform, is_nonlinear, collect_factorizations)
  end
end

function PatchSolver(
  assem::DistributedPatchAssembler, trial::DistributedFESpace, test::DistributedFESpace, weakform::Function; 
  is_nonlinear = false, collect_factorizations = false, assem_kwargs...
)
  if !is_nonlinear
    axes = (get_free_dof_ids(test), get_free_dof_ids(trial))
    patch_axes = map(local_views(assem)) do assem
      assem.strategy.patch_rows, assem.strategy.patch_cols
    end |> tuple_of_arrays
    patch_mats = assemble_matrix(weakform, assem, trial, test)
    return PatchSolverFromMats(axes, patch_axes, patch_mats, collect_factorizations)
  else
    return PatchSolverFromWeakform(assem, trial, test, weakform, is_nonlinear, collect_factorizations)
  end
end

struct PatchSS{A} <: Algebra.SymbolicSetup
  solver::A
end

Algebra.symbolic_setup(s::PatchSolverFromMats,mat::AbstractMatrix) = PatchSS(s)
Algebra.symbolic_setup(s::PatchSolverFromWeakform,mat::AbstractMatrix) = PatchSS(s)

struct PatchNS{A,B,C,D,E} <: Algebra.NumericalSetup
  solver :: A
  patch_rows :: B
  patch_cols :: C
  patch_factorizations :: D
  caches :: E
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromMats},mat::AbstractMatrix)
  solver = ss.solver
  patch_rows, patch_cols = solver.patch_axes
  patch_factorizations = lazy_map(lu!, solver.patch_mats)
  if solver.collect_factorizations
    patch_factorizations = Arrays.lazy_collect(patch_factorizations)
  end

  i_cache = array_cache(patch_cols)
  x_cache = CachedVector(eltype(mat))
  f_cache = array_cache(patch_factorizations)
  caches = (i_cache, x_cache, f_cache)

  return PatchNS(solver, patch_rows, patch_cols, patch_factorizations,caches)
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromMats},mat::PSparseMatrix)
  solver = ss.solver
  # patch_cols = reindex_patch_ids(solver.patch_axes[2], solver.axes[2], axes(mat,2))
  # patch_rows = patch_cols # reindex_patch_ids(solver.patch_axes[1], solver.axes[1], axes(mat,1))
  rows, cols = solver.axes
  patch_rows, patch_cols = solver.patch_axes

  patch_factorizations, caches = map(solver.patch_mats,patch_cols) do patch_mats, patch_cols
    patch_factorizations = lazy_map(lu!, patch_mats)
    if solver.collect_factorizations
      patch_factorizations = Arrays.lazy_collect(patch_factorizations)
    end

    i_cache = array_cache(patch_cols)
    x_cache = CachedVector(eltype(mat))
    f_cache = array_cache(patch_factorizations)
    caches = (i_cache, x_cache, f_cache)

    return patch_factorizations, caches
  end |> tuple_of_arrays

  x_c = pzeros(eltype(mat),partition(rows))
  b_c = pzeros(eltype(mat),partition(cols))
  caches = (x_c,b_c,caches)
  return PatchNS(solver, patch_rows, patch_cols, patch_factorizations, caches)
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromWeakform},mat::AbstractMatrix,vec::AbstractVector)
  solver = ss.solver
  @assert solver.is_nonlinear
  assem, trial, test = solver.assem, solver.trial, solver.test
  
  patch_rows = assem.strategy.patch_rows
  patch_cols = assem.strategy.patch_cols

  xh = FEFunction(trial, vec)
  biform(u,v) = solver.weakform(xh,u,v)
  patch_mats = assemble_matrix(biform, assem, trial, test)
  patch_factorizations = lazy_map(lu!, patch_mats)
  if solver.collect_factorizations
    patch_factorizations = Arrays.lazy_collect(patch_factorizations)
  end

  i_cache = array_cache(patch_cols)
  x_cache = CachedVector(eltype(mat))
  f_cache = array_cache(patch_factorizations)
  caches = (i_cache, x_cache, f_cache)

  return PatchNS(solver, patch_rows, patch_cols, patch_factorizations, caches)
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromWeakform},mat::PSparseMatrix,vec::PVector)
  solver = ss.solver
  @assert solver.is_nonlinear
  assem, trial, test = solver.assem, solver.trial, solver.test
  
  patch_rows = map(a -> a.strategy.patch_rows, local_views(assem))
  patch_cols = map(a -> a.strategy.patch_cols, local_views(assem))
  # patch_cols = reindex_patch_ids(assem_patch_cols, get_free_dof_ids(trial), axes(mat,2))
  # patch_rows = patch_cols # reindex_patch_ids(assem_patch_rows, get_free_dof_ids(test), axes(mat,1))

  xh = FEFunction(trial, vec)
  biform(u,v) = solver.weakform(xh,u,v)
  patch_mats = assemble_matrix(biform, assem, trial, test)
  patch_factorizations, caches = map(patch_mats,patch_cols) do patch_mats,patch_cols
    patch_factorizations = lazy_map(lu!, patch_mats)
    if solver.collect_factorizations
      patch_factorizations = Arrays.lazy_collect(patch_factorizations)
    end

    i_cache = array_cache(patch_cols)
    x_cache = CachedVector(eltype(mat))
    f_cache = array_cache(patch_factorizations)
    caches = (i_cache, x_cache, f_cache)

    return patch_factorizations, caches
  end |> tuple_of_arrays

  x_c = pzeros(eltype(mat),partition(get_free_dof_ids(trial)))
  b_c = pzeros(eltype(mat),partition(get_free_dof_ids(test)))
  caches = (x_c,b_c,caches)
  return PatchNS(solver, patch_rows, patch_cols, patch_factorizations, caches)
end

# function reindex_patch_ids(old_patch_ids, old_ids::PRange, new_ids::PRange)
#   if PartitionedArrays.matching_local_indices(old_ids, new_ids)
#     return old_patch_ids
#   end
#   return map(reindex_patch_ids,old_patch_ids,partition(old_ids),partition(new_ids))
# end
# 
# function reindex_patch_ids(old_patch_ids, old_ids, new_ids)
#   id_map = GridapDistributed.find_local_to_local_map(old_ids, new_ids)
#   new_patch_ids = Arrays.lazy_collect(lazy_map(Broadcasting(Reindex(id_map)),old_patch_ids))
#   @assert !any(ids -> any(iszero,ids), new_patch_ids)
#   return new_patch_ids
# end

function Algebra.solve!(x::PVector,ns::PatchNS,b::PVector)
  _, b_c, _ = ns.caches
  
  copy!(b_c,b)
  consistent!(b_c) |> wait
  patch_b = map(partition(b_c),ns.patch_rows) do b_c,patch_rows
    lazy_map(Broadcasting(Reindex(b_c)),patch_rows)
  end
  solve_patch_overlapping!(x,ns,patch_b)
end

function Algebra.solve!(x::AbstractVector,ns::PatchNS,b::AbstractVector)
  patch_b = lazy_map(Broadcasting(Reindex(b)),ns.patch_rows)
  solve_patch_overlapping!(x,ns,patch_b)
end

for func in (:solve_patch_overlapping!, :solve_patch_nonoverlapping!)
  @eval begin

    function $func(x::AbstractVector,ns::PatchNS,patch_b)
      return $func(x, ns.patch_cols, ns.patch_factorizations, patch_b, eachindex(patch_b), ns.caches)
    end

    function $func(x::PVector,ns::PatchNS,patch_b)
      x_c, _, caches = ns.caches
      map(partition(x_c), ns.patch_cols, ns.patch_factorizations, patch_b, caches) do x_c, patch_cols, patch_f, patch_b, caches
        $func(x_c, patch_cols, patch_f, patch_b, eachindex(patch_b), caches)
      end
      assemble!(x_c) |> wait
      copy!(x, x_c)
      consistent!(x) |> wait
      return x
    end

  end
end

function solve_patch_overlapping!(
  x, patch_cols, patch_f, patch_b,
  patch_ids = eachindex(patch_b),
  caches = (array_cache(patch_cols),CachedVector(eltype(x)),array_cache(patch_f))
)
  i_cache, x_cache, f_cache = caches
  b_cache = array_cache(patch_b)

  fill!(x,zero(eltype(x)))
  for patch in patch_ids
    cols = getindex!(i_cache,patch_cols,patch)
    isempty(cols) && continue
    fp = getindex!(f_cache,patch_f,patch)
    bp = getindex!(b_cache,patch_b,patch)
    setsize!(x_cache,(length(cols),))
    xp = x_cache.array
    ldiv!(xp,fp,bp)
    x[cols] .+= xp
  end

  return x
end

function solve_patch_nonoverlapping!(
  x, patch_cols, patch_f, patch_b,
  patch_ids = eachindex(patch_b),
  caches = (array_cache(patch_cols),CachedVector(eltype(x)),array_cache(patch_f))
)
  i_cache, x_cache, f_cache = caches
  b_cache = array_cache(patch_b)

  fill!(x,zero(eltype(x)))
  for patch in patch_ids
    cols = getindex!(i_cache,patch_cols,patch)
    isempty(cols) && continue
    fp = getindex!(f_cache,patch_f,patch)
    bp = getindex!(b_cache,patch_b,patch)
    xp = view(x,cols)
    ldiv!(xp,fp,bp)
  end

  return x
end
