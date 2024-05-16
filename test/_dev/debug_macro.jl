using PartitionedArrays
using GridapDistributed

macro pdebug(parts,msg)
  return quote
    if i_am_main($parts)
      @debug $msg
    end
  end
end

np = 4
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

@pdebug(parts,"Hello, world!")
