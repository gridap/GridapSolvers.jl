
struct PatchSolver{A,B,C,D,E} <: Algebra.LinearSolver
  space::A
  patch_decomposition::B
  patch_ids::C
  patch_cell_lids::D
  biform::E
end

function PatchSolver(
  space::FESpace,patch_decomposition::PatchDecomposition,biform,reffe;conformity=nothing
)
  cell_conformity = MultilevelTools._cell_conformity(
    get_background_model(get_triangulation(space)),reffe;conformity
  )
  return PatchSolver(space,patch_decomposition,biform,cell_conformity)
end

function PatchSolver(
  space::FESpace,patch_decomposition::PatchDecomposition,biform,cell_conformity::CellConformity
)
  cell_dof_ids = get_cell_dof_ids(space)
  patch_cells = get_patch_cells(patch_decomposition)
  patch_cells_overlapped = get_patch_cells_overlapped(patch_decomposition)

  patch_cell_lids, _ = generate_patch_cell_dofs_ids(
    patch_cells,patch_cells_overlapped,
    patch_decomposition.patch_cells_faces_on_boundary,
    cell_dof_ids,cell_conformity;numbering=:local
  )
  patch_ids = generate_pdof_to_dof(
    patch_decomposition,cell_dof_ids,patch_cell_lids
  )
  return PatchSolver(space,patch_decomposition,patch_ids,patch_cell_lids,biform)
end

function PatchSolver(
  space::MultiFieldFESpace,patch_decomposition::PatchDecomposition,biform,reffes;kwargs...
)
  solvers = map((Si,reffe)->PatchSolver(Si,patch_decomposition,biform,reffe;kwargs...),space,reffes)
  field_to_patch_cell_lids = map(s -> s.patch_cell_lids ,solvers)
  field_to_patch_ids = map(s -> s.patch_ids ,solvers)
  field_to_ndofs = map(s -> num_free_dofs(s), space)

  patch_cell_lids, patch_ids = block_patch_ids(patch_decomposition,field_to_patch_cell_lids,field_to_patch_ids,field_to_ndofs)
  return PatchSolver(space,patch_decomposition,patch_ids,patch_cell_lids,biform)
end

function compute_patch_offsets(
  patch_decomposition::PatchDecomposition,
  field_to_patch_ids::Vector
)
  nfields = length(field_to_patch_ids)
  npatches = num_patches(patch_decomposition)

  offsets = zeros(Int,(nfields,npatches))
  for i in 1:(nfields-1)
    offsets[i+1,:] .= offsets[i,:] .+ map(length,field_to_patch_ids[i])
  end

  return offsets
end

function block_patch_ids(
  patch_decomposition::PatchDecomposition,
  field_to_patch_cell_lids::Vector,
  field_to_patch_ids::Vector,
  field_to_ndofs::Vector
)
  offsets = compute_patch_offsets(patch_decomposition,field_to_patch_ids)
  nfields = length(field_to_patch_ids)
  npatches = num_patches(patch_decomposition)

  field_offsets = zeros(Int,nfields)
  for i in 1:(nfields-1)
    field_offsets[i+1] = field_offsets[i] + field_to_ndofs[i]
  end

  block_cell_lids = Any[]
  block_ids = Any[]
  for i in 1:nfields
    patch_cell_lids_i = field_to_patch_cell_lids[i]
    patch_ids_i = field_to_patch_ids[i]
    if i == 1
      push!(block_cell_lids,patch_cell_lids_i)
      push!(block_ids,patch_ids_i)
    else
      offsets_i = Int32.(offsets[i,:])
      pcell_to_offsets = patch_extend(patch_decomposition,offsets_i)
      patch_cell_lids_i_b = lazy_map(Broadcasting(Gridap.MultiField._sum_if_first_positive),patch_cell_lids_i,pcell_to_offsets)
      patch_ids_i_b = lazy_map(Broadcasting(Gridap.MultiField._sum_if_first_positive),patch_ids_i,Fill(field_offsets[i],npatches))
      push!(block_cell_lids,patch_cell_lids_i_b)
      push!(block_ids,patch_ids_i_b)
    end
  end
  patch_cell_lids = lazy_map(BlockMap(nfields,collect(1:nfields)),block_cell_lids...)
  patch_ids = map(vcat,block_ids...)
  return patch_cell_lids, patch_ids
end

struct PatchSS{A} <: Algebra.SymbolicSetup
  solver::PatchSolver{A}
end

Algebra.symbolic_setup(s::PatchSolver,mat::AbstractMatrix) = PatchSS(s)

struct PatchNS{A,B} <: Algebra.NumericalSetup
  solver::A
  cache ::B
end

function Algebra.numerical_setup(ss::PatchSS,mat::AbstractMatrix)
  T = eltype(mat)

  space = ss.solver.space
  biform = ss.solver.biform
  cellmat, _ = move_contributions( # TODO: Generalise
    get_array(biform(get_trial_fe_basis(space),get_fe_basis(space))),
    Triangulation(ss.solver.patch_decomposition)
  )
  cache = (CachedArray(zeros(T,1)), CachedArray(zeros(T,1,1)), cellmat)
  return PatchNS(ss.solver,cache)
end

function Algebra.solve!(x::AbstractVector,ns::PatchNS,b::AbstractVector)
  PD = ns.solver.patch_decomposition
  patch_ids, patch_cell_lids = ns.solver.patch_ids, ns.solver.patch_cell_lids
  c_xk, c_Ak, cellmat = ns.cache
  fill!(x,0.0)

  rows_cache = array_cache(patch_cell_lids)
  cols_cache = array_cache(patch_cell_lids)
  vals_cache = array_cache(cellmat)

  add! = AddEntriesMap(+)
  add_cache = return_cache(add!,c_Ak.array,first(cellmat),first(patch_cell_lids),first(patch_cell_lids))
  caches = add_cache, vals_cache, rows_cache, cols_cache

  n_patches = length(patch_ids)
  for patch in 1:n_patches
    ids = patch_ids[patch]

    n = length(ids)
    setsize!(c_xk,(n,))
    setsize!(c_Ak,(n,n))
    Ak = c_Ak.array
    bk = c_xk.array

    patch_cellmat = patch_view(PD,cellmat,patch)
    patch_rows = patch_view(PD,patch_cell_lids,patch)
    patch_cols = patch_view(PD,patch_cell_lids,patch)
    fill!(Ak,0.0)
    FESpaces._numeric_loop_matrix!(Ak,caches,patch_cellmat,patch_rows,patch_cols)

    copyto!(bk,view(b,ids))
    f = lu!(Ak;check=false)
    @check issuccess(f) "Factorization failed"
    ldiv!(f,bk)
    
    x[ids] .+= bk
  end

  return x
end
