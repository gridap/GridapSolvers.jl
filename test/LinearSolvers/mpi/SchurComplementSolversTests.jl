module SchurComplementSolversTestsMPI
using PartitionedArrays, MPI
include("../SchurComplementSolversTests.jl")

with_mpi() do distribute
  SchurComplementSolversTests.main(distribute,(2,2))
end

end