
function generate_subparts(root_parts::AbstractPData,subpart_size::Integer)
  root_comm = root_parts.comm
  rank = MPI.Comm_rank(root_comm)
  size = MPI.Comm_size(root_comm)
  Gridap.Helpers.@check all(subpart_size .<= size)
  Gridap.Helpers.@check all(subpart_size .>= 1)

  if rank < subpart_size
    comm = MPI.Comm_split(root_comm, 0, 0)
  else
    comm = MPI.Comm_split(root_comm, MPI.MPI_UNDEFINED, MPI.MPI_UNDEFINED)
  end
  return get_part_ids(comm)
end

function generate_level_parts(root_parts::AbstractPData,last_level_parts::AbstractPData,level_parts_size::Integer)
  if level_parts_size == num_parts(last_level_parts)
    return last_level_parts
  end
  return generate_subparts(root_parts,level_parts_size)
end

function generate_level_parts(root_parts::AbstractPData,num_procs_x_level::Vector{<:Integer})
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{typeof(parts)}(undef,num_levels)
  level_parts[1] = generate_subparts(root_parts,num_procs_x_level[1])
  for l = 2:num_levels
    level_parts[l] = generate_level_parts(root_parts,level_parts[l-1],num_procs_x_level[l])
  end
  return level_parts
end