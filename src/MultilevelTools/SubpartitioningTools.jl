
"""
    generate_level_parts(root_parts::AbstractArray,num_procs_x_level::Vector{<:Integer})
  
  From a root communicator `root_parts`, generate a sequence of nested 
  subcommunicators with sizes given by `num_procs_x_level`.
"""
function generate_level_parts(root_parts::AbstractArray,num_procs_x_level::Vector{<:Integer})
  num_levels = length(num_procs_x_level)
  T = Union{typeof(root_parts),GridapDistributed.MPIVoidVector{eltype(root_parts)}}
  level_parts = Vector{T}(undef,num_levels)
  level_parts[1] = generate_subparts(root_parts,num_procs_x_level[1])
  for l = 2:num_levels
    if num_procs_x_level[l] == num_procs_x_level[l-1]
      level_parts[l] = level_parts[l-1]
    else
      level_parts[l] = generate_subparts(root_parts, num_procs_x_level[l])
    end
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