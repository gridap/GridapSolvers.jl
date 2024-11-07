module NonlinearSolversTestsMPI
using MPI, PartitionedArrays
include("../NonlinearSolversTests.jl")

with_mpi() do distribute
  NonlinearSolversTests.main(distribute,4,:newton)
  NonlinearSolversTests.main(distribute,4,:newton_continuation)
end

end