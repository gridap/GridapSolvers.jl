
function num_parts(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    nparts = MPI.Comm_size(comm)
  else
    nparts = -1
  end
  nparts
end

num_parts(comm::MPIArray) = num_parts(comm.comm)
num_parts(comm::GridapDistributed.MPIVoidVector) = num_parts(comm.comm)

function get_part_id(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    id = MPI.Comm_rank(comm)+1
  else
    id = -1
  end
  id
end

i_am_in(comm::MPI.Comm) = get_part_id(comm) >=0
i_am_in(comm::MPIArray) = i_am_in(comm.comm)
i_am_in(comm::GridapDistributed.MPIVoidVector) = i_am_in(comm.comm)

function generate_level_parts(root_parts::AbstractArray,last_level_parts::AbstractArray,level_parts_size::Integer)
  if level_parts_size == num_parts(last_level_parts)
    return last_level_parts
  end
  return generate_subparts(root_parts,level_parts_size)
end

function generate_level_parts(root_parts::AbstractArray,num_procs_x_level::Vector{<:Integer})
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{typeof(parts)}(undef,num_levels)
  level_parts[1] = generate_subparts(root_parts,num_procs_x_level[1])
  for l = 2:num_levels
    level_parts[l] = generate_level_parts(root_parts,level_parts[l-1],num_procs_x_level[l])
  end
  return level_parts
end

my_print(x::PVector,s) = my_print(partition(x),s)

function my_print(x::MPIArray,s)
  parts = linear_indices(x)
  i_am_main(parts) && println(s)
  map(parts,x) do p,xi
    sleep(p*0.2)
    println("   > $p: ", xi)
  end
  sleep(2)
end