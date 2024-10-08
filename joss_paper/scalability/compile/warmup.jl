
using Scalability

stokes_main(;
  nr=1,
  np=(1,1),
  nc=(4,4),
  title="data/compile",
)

stokes_gmg_main(;
  nr=1,
  np=(1,1),
  nc=(4,4),
  np_per_level=[(1,1),(1,1)],
  title="data/compile_gmg",
  mode = :mpi
)
