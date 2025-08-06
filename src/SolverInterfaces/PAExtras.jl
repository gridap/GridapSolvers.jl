
function split_indices(ids)
  OwnAndGhostIndices(
    OwnIndices(global_length(ids), part_id(ids), own_to_global(ids)),
    GhostIndices(global_length(ids), ghost_to_global(ids), ghost_to_owner(ids))
  )
end

function fetch_ghost_rows(A::PSparseMatrix,new_rows)
  @assert PartitionedArrays.matching_own_indices(axes(A)...)
  @assert reduce(&,map((a,b) -> own_to_local(a) == own_to_local(b),partition(axes(A,1)),new_rows);init=true)
  mats = partition(A)
  rows = partition(axes(A,1))
  cols = partition(axes(A,2))

  nbors_snd, nbors_rcv = assembly_neighbors(new_rows)
  lids_rcv, lids_snd = assembly_local_indices(new_rows, nbors_snd, nbors_rcv) # Reversed caches
  graph = ExchangeGraph(nbors_snd,nbors_rcv)

  I_snd, J_snd, Jo_snd, V_snd = map(mats, nbors_snd, lids_snd, rows, cols) do mat, nbors_snd, lids_snd, rows, cols
    
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
  end |> tuple_of_arrays

  # Exchange the data
  t_I = exchange(I_snd,graph)
  t_J = exchange(J_snd,graph)
  t_Jo = exchange(Jo_snd,graph)
  t_V = exchange(V_snd,graph)

  I_rcv = fetch(t_I)
  J_rcv = fetch(t_J)
  Jo_rcv = fetch(t_Jo)
  V_rcv = fetch(t_V)

  new_mats, new_cols = map(I_rcv,J_rcv,Jo_rcv,V_rcv,new_rows,cols,mats) do I_rcv, J_rcv, Jo_rcv, V_rcv, new_rows, cols, mat
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
    m_own, m_ghost = own_length(new_rows), ghost_length(new_rows)
    n_own, n_ghost = own_length(new_cols), ghost_length(new_cols)
    mat_own_loc = sparse_hcat(view(mat,1:m_own,:), spzeros(eltype(mat),m_own,n_ghost-ghost_length(cols)))
    mat_ghost_loc = sparse(I_rcv.data, J_rcv.data, V_rcv.data, m_ghost, n_own+n_ghost)
    new_mat = sparse_vcat(mat_own_loc, mat_ghost_loc)

    return new_mat, new_cols
  end |> tuple_of_arrays

  return PSparseMatrix(new_mats, new_rows, new_cols)
end
