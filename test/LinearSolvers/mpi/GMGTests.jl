module GMGTestsMPI
using MPI, PartitionedArrays
include("../GMGTests.jl")

with_mpi() do distribute
  GMGTests.main(distribute,4,(4,4),[(2,2),(2,1),(1,1)]) # 2D
  GMGTests.main(distribute,4,(2,2,2),[(2,2,1),(2,1,1),(1,1,1)]) # 3D
end

end