function _allocate_cell_wise_dofs(cell_to_ldofs)
  map_parts(cell_to_ldofs) do cell_to_ldofs
    cache  = array_cache(cell_to_ldofs)
    ncells = length(cell_to_ldofs)
    ptrs   = Vector{Int32}(undef,ncells+1)
    for cell in 1:ncells
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      ptrs[cell+1] = length(ldofs)
    end
    PArrays.length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data  = Vector{Float64}(undef,ndata)
    PArrays.Table(data,ptrs)
  end
end

function _update_cell_dof_values_with_local_info!(cell_dof_values_new,
                                                  cell_dof_values_old,
                                                  new2old)
   map_parts(cell_dof_values_new,
             cell_dof_values_old,
             new2old) do cell_dof_values_new,cell_dof_values_old,new2old
    ocache = array_cache(cell_dof_values_old)
    for (ncell,ocell) in enumerate(new2old)
      if ocell!=0
        # Copy ocell to ncell
        oentry = getindex!(ocache,cell_dof_values_old,ocell)
        range  = cell_dof_values_new.ptrs[ncell]:cell_dof_values_new.ptrs[ncell+1]-1
        cell_dof_values_new.data[range] .= oentry
      end
    end
   end
end

function allocate_comm_data(num_dofs_x_cell,lids)
  map_parts(num_dofs_x_cell,lids) do num_dofs_x_cell,lids
    n = length(lids)
    ptrs = Vector{Int32}(undef,n+1)
    ptrs.= 0
    for i = 1:n
      for j = lids.ptrs[i]:lids.ptrs[i+1]-1
        ptrs[i+1] = ptrs[i+1] + num_dofs_x_cell.data[j]
      end
    end
    PArrays.length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data  = Vector{Float64}(undef,ndata)
    PArrays.Table(data,ptrs)
  end
end

function pack_snd_data!(snd_data,cell_dof_values,snd_lids)
  map_parts(snd_data,cell_dof_values,snd_lids) do snd_data,cell_dof_values,snd_lids
    cache = array_cache(cell_dof_values)
    s = 1
    for i = 1:length(snd_lids)
      for j = snd_lids.ptrs[i]:snd_lids.ptrs[i+1]-1
        cell  = snd_lids.data[j]
        ldofs = getindex!(cache,cell_dof_values,cell)

        e = s+length(ldofs)-1
        range = s:e
        snd_data.data[range] .= ldofs
        s = e+1
      end
    end
  end
end

function unpack_rcv_data!(cell_dof_values,rcv_data,rcv_lids)
  map_parts(cell_dof_values,rcv_data,rcv_lids) do cell_dof_values,rcv_data,rcv_lids
    s = 1
    for i = 1:length(rcv_lids.ptrs)-1
      for j = rcv_lids.ptrs[i]:rcv_lids.ptrs[i+1]-1
        cell = rcv_lids.data[j]
        range_cell_dof_values = cell_dof_values.ptrs[cell]:cell_dof_values.ptrs[cell+1]-1
        
        e = s+length(range_cell_dof_values)-1
        range_rcv_data = s:e
        cell_dof_values.data[range_cell_dof_values] .= rcv_data.data[range_rcv_data]
        s = e+1
      end
    end
  end
end

function get_glue_components(glue::GridapDistributed.RedistributeGlue,reverse::Val{false})
  return glue.lids_rcv, glue.lids_snd, glue.parts_rcv, glue.parts_snd, glue.new2old
end

function get_glue_components(glue::GridapDistributed.RedistributeGlue,reverse::Val{true})
  return glue.lids_snd, glue.lids_rcv, glue.parts_snd, glue.parts_rcv, glue.old2new
end

function num_dofs_x_cell(cell_dofs_array,lids)
  map_parts(cell_dofs_array,lids) do cell_dofs_array, lids
     data = [length(cell_dofs_array[i]) for i = 1:length(cell_dofs_array) ]
     PArrays.Table(data,lids.ptrs)
  end
end


function redistribute_cell_dofs(cell_dof_values_old::GridapDistributed.DistributedCellDatum,
                                Uh_new::GridapDistributed.DistributedSingleFieldFESpace,
                                model_new,
                                glue::GridapDistributed.RedistributeGlue;
                                reverse=false)

  lids_rcv, lids_snd, parts_rcv, parts_snd, new2old = get_glue_components(glue,Val(reverse))
  cell_dof_ids_new    = map_parts(get_cell_dof_ids, Uh_new.spaces)

  num_dofs_x_cell_snd = num_dofs_x_cell(cell_dof_values_old, lids_snd)
  num_dofs_x_cell_rcv = num_dofs_x_cell(cell_dof_ids_new, lids_rcv)
  snd_data = allocate_comm_data(num_dofs_x_cell_snd, lids_snd)
  rcv_data = allocate_comm_data(num_dofs_x_cell_rcv, lids_rcv)

  pack_snd_data!(snd_data,cell_dof_values_old,lids_snd)

  tout = async_exchange!(rcv_data,
                        snd_data,
                        parts_rcv,
                        parts_snd,
                        PArrays._empty_tasks(parts_rcv))
  map_parts(schedule,tout)

  cell_dof_values_new = _allocate_cell_wise_dofs(cell_dof_ids_new)

  # We have to build the owned part of "cell_dof_values_new" out of
  #  1. cell_dof_values_old (for those cells s.t. new2old[:]!=0)
  #  2. cell_dof_values_new_rcv (for those cells s.t. new2old[:]=0)
  _update_cell_dof_values_with_local_info!(cell_dof_values_new,
                                           cell_dof_values_old,
                                           new2old)

  map_parts(wait,tout)
  unpack_rcv_data!(cell_dof_values_new,rcv_data,lids_rcv)

  # Why are we exchanging something that has already been exchanged?
  fgids = get_cell_gids(model_new)
  exchange!(cell_dof_values_new,fgids.exchanger) 

  return cell_dof_values_new
end

function redistribute_free_values!(fv_new::PVector,
                                  Uh_new::GridapDistributed.DistributedSingleFieldFESpace,
                                  fv_old::PVector,
                                  Uh_old::GridapDistributed.DistributedSingleFieldFESpace,
                                  model_new,
                                  glue::GridapDistributed.RedistributeGlue;
                                  reverse=false)

  uh_old = FEFunction(Uh_old,fv_old)
  cell_dof_values_old = map_parts(get_cell_dof_values,uh_old.fields)
  cell_dof_values_new = redistribute_cell_dofs(cell_dof_values_old,Uh_new,model_new,glue;reverse=reverse)

  # Assemble the new FEFunction
  Gridap.FESpaces.gather_free_values!(fv_new, Uh_new.spaces,cell_dof_values_new)
  return fv_new
end


function redistribute_fe_function(uh_old::GridapDistributed.DistributedSingleFieldFEFunction,
                                  Uh_new::GridapDistributed.DistributedSingleFieldFESpace,
                                  model_new,
                                  glue::GridapDistributed.RedistributeGlue;
                                  reverse=false)

  cell_dof_values_old = map_parts(get_cell_dof_values,uh_old.fields)
  cell_dof_values_new = redistribute_cell_dofs(cell_dof_values_old,Uh_new,model_new,glue;reverse=reverse)

  # Assemble the new FEFunction
  free_values, dirichlet_values = Gridap.FESpaces.gather_free_and_dirichlet_values(Uh_new.spaces,cell_dof_values_new)
  free_values = PVector(free_values,Uh_new.gids)
  uh_new = FEFunction(Uh_new,free_values,dirichlet_values)
  return uh_new
end


function Gridap.FESpaces.gather_free_and_dirichlet_values(f::Gridap.Distributed.AbstractDistributedFESpace,cv)
  free_values, dirichlet_values = map_parts(local_views(f),cv) do f, cv
    Gridap.FESpaces.gather_free_and_dirichlet_values(f,cv)
  end
  return free_values, dirichlet_values
end