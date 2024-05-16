
"""
    generate_level_parts(root_parts::AbstractArray,num_procs_x_level::Vector{<:Integer})
  
  From a root communicator `root_parts`, generate a sequence of nested 
  subcommunicators with sizes given by `num_procs_x_level`.
"""
function generate_level_parts(root_parts::AbstractArray,num_procs_x_level::Vector{<:Integer})
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{typeof(parts)}(undef,num_levels)
  level_parts[1] = generate_subparts(root_parts,num_procs_x_level[1])
  for l = 2:num_levels
    level_parts[l] = generate_level_parts(root_parts,level_parts[l-1],num_procs_x_level[l])
  end
  return level_parts
end

function generate_level_parts(root_parts::AbstractArray,last_level_parts::AbstractArray,level_parts_size::Integer)
  if level_parts_size == num_parts(last_level_parts)
    return last_level_parts
  end
  return generate_subparts(root_parts,level_parts_size)
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