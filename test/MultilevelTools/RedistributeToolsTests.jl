module RedistributeToolsTests
using MPI
using PartitionedArrays
using Gridap, Gridap.Algebra
using GridapDistributed
using Test

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapDistributed: redistribute_cell_dofs, redistribute_fe_function, redistribute_free_values

function get_model_hierarchy(parts,Dc,np_per_level)
  domain = ifelse(Dc == 2, (0,1,0,1), (0,1,0,1,0,1))
  nc = ifelse(Dc == 2, (4,4), (4,4,4))
  mh = CartesianModelHierarchy(parts,np_per_level,domain,nc)
  return mh
end

function main_driver(parts,mh)
  level_parts = get_level_parts(mh)
  old_parts = level_parts[2]
  new_parts = level_parts[1]

  # FE Spaces
  order = 2
  u(x)  = x[1]^2 + x[2]^2 - 3.0*x[1]*x[2]
  reffe = ReferenceFE(lagrangian,Float64,order)
  glue  = mh[1].red_glue

  model_old = get_model_before_redist(mh[1])
  if i_am_in(old_parts)
    VOLD  = TestFESpace(model_old,reffe,dirichlet_tags="boundary")
    UOLD  = TrialFESpace(VOLD,u)
  else
    VOLD  = nothing
    UOLD  = nothing
  end

  model_new = get_model(mh[1])
  VNEW  = TestFESpace(model_new,reffe,dirichlet_tags="boundary")
  UNEW  = TrialFESpace(VNEW,u)

  # Triangulations
  qdegree = 2*order+1
  Ω_new   = Triangulation(model_new)
  dΩ_new  = Measure(Ω_new,qdegree)
  uh_new  = interpolate(u,UNEW)

  if i_am_in(old_parts)
    Ω_old   = Triangulation(model_old)
    dΩ_old  = Measure(Ω_old,qdegree)
    uh_old  = interpolate(u,UOLD)
  else
    Ω_old   = nothing
    dΩ_old  = nothing
    uh_old  = nothing
  end

  # Old -> New
  uh_old_red = redistribute_fe_function(uh_old,UNEW,model_new,glue)
  n = sum(∫(uh_old_red)*dΩ_new)
  if i_am_in(old_parts)
    o = sum(∫(uh_old)*dΩ_old)
    @test o ≈ n
  end

  # New -> Old
  uh_new_red = redistribute_fe_function(uh_new,UOLD,model_old,glue;reverse=true)
  n = sum(∫(uh_new)*dΩ_new)
  if i_am_in(old_parts)
    o = sum(∫(uh_new_red)*dΩ_old)
    @test o ≈ n
  end
end

function main(distribute,np,Dc,np_x_level)
  parts = distribute(LinearIndices((np,)))
  mh = get_model_hierarchy(parts,Dc,np_x_level)
  main_driver(parts,mh)
end

end # module RedistributeToolsTests