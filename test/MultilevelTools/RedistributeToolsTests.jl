module RedistributeToolsTests
using MPI
using PartitionedArrays
using Gridap, Gridap.Algebra
using GridapDistributed
using GridapP4est
using Test

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapDistributed: redistribute_cell_dofs, redistribute_fe_function, redistribute_free_values

function get_model_hierarchy(parts,Dc,num_parts_x_level)
  mh = GridapP4est.with(parts) do
    if Dc == 2
      domain = (0,1,0,1)
      nc = (2,2)
    else
      @assert Dc == 3
      domain = (0,1,0,1,0,1)
      nc = (2,2,2)
    end
    num_refs_coarse = 2
    num_levels   = length(num_parts_x_level)
    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,nc)
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
    return ModelHierarchy(parts,coarse_model,num_parts_x_level)
  end
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
  glue  = mh.levels[1].red_glue

  model_old = get_model_before_redist(mh.levels[1])
  if i_am_in(old_parts)
    VOLD  = TestFESpace(model_old,reffe,dirichlet_tags="boundary")
    UOLD  = TrialFESpace(VOLD,u)
  else
    VOLD  = nothing
    UOLD  = nothing
  end

  model_new = get_model(mh.levels[1])
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