
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

mutable struct PatchNS{A,B,C} <: Algebra.NumericalSetup
  solver :: A
  patch_rows :: B
  patch_cols :: C
  patch_mats
  patch_factorizations
  caches
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromMats},mat::AbstractMatrix)
  solver = ss.solver
  patch_rows, patch_cols = solver.patch_axes
  patch_mats = solver.patch_mats
  
  patch_factorizations, caches = patch_solver_caches(
    patch_cols, patch_mats; collect_factorizations = solver.collect_factorizations
  )
  return PatchNS(solver, patch_rows, patch_cols, patch_mats, patch_factorizations,caches)
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromMats},mat::PSparseMatrix)
  solver = ss.solver
  rows, cols = solver.axes
  patch_rows, patch_cols = solver.patch_axes
  patch_mats = solver.patch_mats

  patch_factorizations, caches = map(patch_mats,patch_cols) do patch_mats, patch_cols
    patch_solver_caches(
      patch_cols, patch_mats; collect_factorizations = solver.collect_factorizations
    )
  end |> tuple_of_arrays

  x_c = pzeros(eltype(mat),partition(rows))
  b_c = pzeros(eltype(mat),partition(cols))
  caches = (x_c,b_c,caches)
  return PatchNS(solver, patch_rows, patch_cols, patch_mats, patch_factorizations, caches)
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromWeakform},mat::AbstractMatrix,vec::AbstractVector)
  solver = ss.solver
  @assert solver.is_nonlinear
  assem, trial, test = solver.assem, solver.trial, solver.test

  xh = FEFunction(trial, vec)
  biform(u,v) = solver.weakform(xh,u,v)
  patch_mats = assemble_matrix(biform, assem, trial, test)
  patch_rows, patch_cols, patch_factorizations, caches = patch_solver_caches(
    assem, patch_mats; collect_factorizations = solver.collect_factorizations
  )

  return PatchNS(solver, patch_rows, patch_cols, patch_mats, patch_factorizations, caches)
end

function Algebra.numerical_setup(ss::PatchSS{<:PatchSolverFromWeakform},mat::PSparseMatrix,vec::PVector)
  solver = ss.solver
  @assert solver.is_nonlinear
  assem, trial, test = solver.assem, solver.trial, solver.test

  xh = FEFunction(trial, vec)
  biform(u,v) = solver.weakform(xh,u,v)
  patch_mats = assemble_matrix(biform, assem, trial, test)

  patch_rows, patch_cols, patch_factorizations, caches = patch_solver_caches(
    assem, patch_mats; collect_factorizations = solver.collect_factorizations
  )

  x_c = pzeros(eltype(mat),partition(get_free_dof_ids(trial)))
  b_c = pzeros(eltype(mat),partition(get_free_dof_ids(test)))
  caches = (x_c,b_c,caches)
  return PatchNS(solver, patch_rows, patch_cols, patch_mats, patch_factorizations, caches)
end

function Algebra.numerical_setup!(ns::PatchNS,mat::AbstractMatrix,vec::AbstractVector)
  solver = nc.solver
  @assert solver.is_nonlinear
  assem, trial, test = solver.assem, solver.trial, solver.test

  xh = FEFunction(trial, vec)
  biform(u,v) = solver.weakform(xh,u,v)
  patch_mats = assemble_matrix(biform, assem, trial, test)
  patch_factorizations, caches = patch_solver_caches(
    patch_mats, ns.patch_cols; collect_factorizations = solver.collect_factorizations
  )
  ns.patch_mats = patch_mats
  ns.patch_factorizations = patch_factorizations
  ns.caches = caches
  
  return ns
end

function Algebra.numerical_setup!(ns::PatchNS,mat::PSparseMatrix,vec::PVector)
  solver = ns.solver
  @assert solver.is_nonlinear
  assem, trial, test = solver.assem, solver.trial, solver.test

  xh = FEFunction(trial, vec)
  biform(u,v) = solver.weakform(xh,u,v)
  patch_mats = assemble_matrix(biform, assem, trial, test)
  patch_factorizations, caches = map(patch_mats,ns.patch_cols) do patch_mats, patch_cols
    patch_solver_caches(patch_cols, patch_mats; collect_factorizations = solver.collect_factorizations)
  end |> tuple_of_arrays

  x_c, b_c, _ = ns.caches
  caches = (x_c, b_c, caches)

  ns.patch_mats = patch_mats
  ns.patch_factorizations = patch_factorizations
  ns.caches = caches
  
  return ns
end

function patch_solver_caches(patch_cols, patch_mats; collect_factorizations = false)
  patch_factorizations = lazy_map(lu!, patch_mats)
  if collect_factorizations
    patch_factorizations = Arrays.lazy_collect(patch_factorizations)
  end

  T = eltype(eltype(patch_mats))
  i_cache = array_cache(patch_cols)
  x_cache = CachedVector(eltype(T))
  f_cache = array_cache(patch_factorizations)
  caches = (i_cache, x_cache, f_cache)

  return patch_factorizations, caches
end

function patch_solver_caches(
  assem::PatchAssembler, patch_mats; 
  collect_factorizations = false
)
  patch_cols = assem.strategy.patch_cols
  patch_rows = assem.strategy.patch_rows

  patch_factorizations, caches = patch_solver_caches(
    patch_cols, patch_mats; collect_factorizations
  )
  return patch_cols, patch_rows, patch_factorizations, caches
end

function patch_solver_caches(
  assem::GridapDistributed.DistributedPatchAssembler, patch_mats; 
  collect_factorizations = false
)
  patch_rows, patch_cols, patch_factorizations, caches = map(local_views(assem), patch_mats) do assem, patch_mats
    patch_solver_caches(assem, patch_mats; collect_factorizations)
  end |> tuple_of_arrays
  return patch_rows, patch_cols, patch_factorizations, caches
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

function solve_patch_overlapping!(x::AbstractVector,ns::PatchNS,patch_b,patch_ids=eachindex(patch_b))
  solve_patch_overlapping!(x, ns.patch_cols, ns.patch_factorizations, patch_b, patch_ids, ns.caches)
end

function solve_patch_nonoverlapping!(x::AbstractVector,ns::PatchNS,patch_b,patch_ids=eachindex(patch_b))
  solve_patch_nonoverlapping!(x, ns.patch_cols, ns.patch_factorizations, patch_b, patch_ids, ns.caches)
end

function solve_patch_overlapping!(x::PVector,ns::PatchNS,patch_b,patch_ids=map(eachindex,patch_b))
  x_c, _, caches = ns.caches
  map(solve_patch_overlapping!, partition(x_c), ns.patch_cols, ns.patch_factorizations, patch_b, patch_ids, caches)
  assemble!(x_c) |> wait
  copy!(x, x_c)
  consistent!(x) |> wait
  return x
end

function solve_patch_nonoverlapping!(x::PVector,ns::PatchNS,patch_b,patch_ids=map(eachindex,patch_b))
  x_c, _, caches = ns.caches
  map(solve_patch_nonoverlapping!, partition(x_c), ns.patch_cols, ns.patch_factorizations, patch_b, patch_ids, caches)
  assemble!(x_c) |> wait
  copy!(x, x_c)
  consistent!(x) |> wait
  return x
end

function solve_patch_overlapping!(x::PVector, args...)
  map(solve_patch_overlapping!, partition(x), args...)
  assemble!(x) |> wait
end

function solve_patch_nonoverlapping!(x::PVector, args...)
  map(solve_patch_nonoverlapping!, partition(x), args...)
  assemble!(x) |> wait
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

function patch_evaluate_rhs(ns::PatchNS,x::AbstractVector)
  patch_x = lazy_map(Broadcasting(Reindex(x)),ns.patch_cols)
  patch_b = lazy_map(*,ns.patch_mats,patch_x)
  return patch_b
end

function patch_evaluate_rhs(ns::PatchNS,x::PVector)
  patch_b = map(partition(x),ns.patch_cols,ns.patch_mats) do x,patch_cols,patch_mats
    patch_x = lazy_map(Broadcasting(Reindex(x)),patch_cols)
    return lazy_map(*,patch_mats,patch_x)
  end
  return patch_b
end
