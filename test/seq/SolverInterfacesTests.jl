
using GridapSolvers
import GridapSolvers.SolverInterfaces as SI
using GridapSolvers.LinearSolvers

info = SI.SolverInfo("my_solver")
SI.log_convergence_info!(info, 10, 1e-6, 1e-5)

summary(info)
summary(info.tols)