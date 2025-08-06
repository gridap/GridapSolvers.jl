
function split_indices(ids)
  OwnAndGhostIndices(
    OwnIndices(global_length(ids), part_id(ids), own_to_global(ids)),
    GhostIndices(global_length(ids), ghost_to_global(ids), ghost_to_owner(ids))
  )
end

function fetch_ghost_rows(A::PSparseMatrix,new_rows)
  @assert PartitionedArrays.matching_own_indices(axes(A)...)
  @assert reduce(&,map((a,b) -> own_to_local(a) == own_to_local(b),partition(axes(A,1)),new_rows);init=true)

  function setup_snd(mat, nbors_snd, lids_snd, rows, cols)
    # Invert the nbor_to_lids table
    n_own = own_length(rows)
    lid_to_nbors = JaggedArray(Arrays.inverse_table(lids_snd.data,lids_snd.ptrs,n_own)...)

    # Count the data to send to each neighbor
    ptrs = zeros(Int32,length(nbors_snd)+1)
    for (i,j,v) in nziterator(mat)
      (i > n_own) && continue # Skip the ghost rows
      for nb in lid_to_nbors[i]
        ptrs[nb+1] += 1
      end
    end
    Arrays.length_to_ptrs!(ptrs)

    # Collect data to send
    n_data = ptrs[end]-1
    I_snd = JaggedArray(zeros(Int,n_data),ptrs)
    J_snd = JaggedArray(zeros(Int,n_data),ptrs)
    Jo_snd = JaggedArray(zeros(Int32,n_data),ptrs)
    V_snd = JaggedArray(zeros(eltype(mat),n_data),ptrs)
    loc_to_glob_rows = local_to_global(rows)
    loc_to_glob_cols = local_to_global(cols)
    loc_to_owner_cols = local_to_owner(cols)
    for (i,j,v) in nziterator(mat)
      (i > n_own) && continue # Skip the ghost rows
      for nb in lid_to_nbors[i]
        k = ptrs[nb]
        I_snd.data[k] = loc_to_glob_rows[i]
        J_snd.data[k] = loc_to_glob_cols[j]
        Jo_snd.data[k] = loc_to_owner_cols[j]
        V_snd.data[k] = v
        ptrs[nb] += 1
      end
    end
    Arrays.rewind_ptrs!(ptrs)

    return I_snd, J_snd, Jo_snd, V_snd
  end

  function process_rcv(mat, I_rcv, J_rcv, Jo_rcv, V_rcv, new_rows, cols)
    # Find new colum range
    # New column indices get appended to the end of the range
    new_cols = union_ghost(cols, J_rcv.data, Jo_rcv.data)

    # Map to new local indices
    glob_to_ghost_rows = global_to_ghost(new_rows)
    glob_to_loc_cols = global_to_local(new_cols)
    for k in eachindex(I_rcv.data)
      I_rcv.data[k] = glob_to_ghost_rows[I_rcv.data[k]]
      J_rcv.data[k] = glob_to_loc_cols[J_rcv.data[k]]
      @check !iszero(I_rcv.data[k])
    end

    # Create new matrix by appending ghost rows
    m = ghost_length(new_rows)
    n = local_length(new_cols)
    if typeof(mat) <: SparseMatrixCSC
      mat_ghost_loc = sparse(I_rcv.data, J_rcv.data, V_rcv.data, m, n)
    elseif typeof(mat) <: SparseMatrixCSR
      Bi = SparseMatricesCSR.getBi(mat)
      mat_ghost_loc = sparsecsr(Val(Bi), I_rcv.data, J_rcv.data, V_rcv.data, m, n)
    else
      @error "Unsupported matrix type: $(typeof(mat))"
    end
    return mat_ghost_loc, new_cols
  end

  mats = partition(A)
  rows = partition(axes(A,1))
  cols = partition(axes(A,2))
  nbors_snd, nbors_rcv = assembly_neighbors(new_rows)
  lids_rcv, lids_snd = assembly_local_indices(new_rows, nbors_snd, nbors_rcv) # Reversed caches

  # Prepare the data to send
  I_snd, J_snd, Jo_snd, V_snd = map(
    setup_snd, mats, nbors_snd, lids_snd, rows, cols
  ) |> tuple_of_arrays

  # Exchange the data
  graph = ExchangeGraph(nbors_snd,nbors_rcv)
  t_J, t_Jo = exchange(J_snd,graph), exchange(Jo_snd,graph)
  t_I, t_V = exchange(I_snd,graph), exchange(V_snd,graph)
  J_rcv, Jo_rcv = fetch(t_J), fetch(t_Jo)
  I_rcv, V_rcv = fetch(t_I), fetch(t_V)

  # Build the new matrix
  mats_ghost_local, new_cols = map(
    process_rcv, mats, I_rcv, J_rcv, Jo_rcv, V_rcv, new_rows, cols
  ) |> tuple_of_arrays
  mats_own_local = own_local_values_view(A, new_cols)
  new_mats = map(mats_own_local, mats_ghost_local) do ol, gl
    # mortar(reshape([ol, gl],(2,1)))
    sparse_vcat(ol,gl)
  end

  return PSparseMatrix(new_mats, new_rows, new_cols)
end

# In the following functions, we assume:
#  - The column range of A is a subset of new_cols, and the 
#    old columns appear in the same order
#  - A potentially contains ghost rows, which we want to remove
# In all cases, we try to alias the data as much as possible.
function own_local_values_view(A::PSparseMatrix, new_cols)
  MT = eltype(partition(A))
  own_local_values_view(MT,A::PSparseMatrix, new_cols)
end

function own_local_values_view(::Type{<:SparseMatrixCSC},A::PSparseMatrix, new_cols)
  rows, cols = axes(A)
  map(partition(A),partition(rows),partition(cols),new_cols) do A, rows, cols, new_cols
    m, colptr, rowval, nzval = A.m, A.colptr, A.rowval, A.nzval
    
    n_new = length(new_cols)
    old_to_new_cols = GridapDistributed.find_local_to_local_map(cols, new_cols)
    colptr_new = zeros(eltype(colptr), n_new + 1)
    for (old, new) in enumerate(old_to_new_cols)
      colptr_new[new+1] = length(nzrange(A, old))
    end
    Arrays.length_to_ptrs!(colptr_new)
    @check colptr_new[end]-1 == length(nzval)
    
    return PartitionedArrays.SubSparseMatrix(
      SparseMatrixCSC(m, n_new, colptr_new, rowval, nzval),
      (own_to_local(rows), Base.OneTo(n_new)), # Sub-indices
      (local_to_own(rows), Base.OneTo(n_new))  # Inverse sub-indices
    )
  end
end

function own_local_values_view(::Type{<:SparseMatrixCSR{Bi}},A::PSparseMatrix, new_cols) where Bi
  rows, cols = axes(A)
  map(partition(A),partition(rows),partition(cols),new_cols) do A, rows, cols, new_cols
    m, rowptr, colval, nzval = A.n, A.rowptr, A.colval, A.nzval

    n_new = length(new_cols)
    old_to_new_cols = GridapDistributed.find_local_to_local_map(cols, new_cols)
    if old_to_new_cols == 1:length(cols) # Old cols are not permuted
      A_new = A
    else
      o = Bi-1
      Ti = eltype(colval)
      new_colval = [Ti(old_to_new_cols[c-o]+o) for c in colval]
      A_new = SparseMatrixCSR{Bi}(
        m, n_new, rowptr, new_colval, nzval
      )
    end

    return PartitionedArrays.SubSparseMatrix(
      A_new, 
      (own_to_local(rows), Base.OneTo(n_new)), # Sub-indices
      (local_to_own(rows), Base.OneTo(n_new))  # Inverse sub-indices
    )
  end
end
