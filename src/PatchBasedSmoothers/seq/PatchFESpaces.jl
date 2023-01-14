struct PatchFESpace  <: Gridap.FESpaces.SingleFieldFESpace
  num_dofs::Int
  patch_cell_dofs_ids::Gridap.Arrays.Table
  Vh::Gridap.FESpaces.SingleFieldFESpace
  patch_decomposition::PatchDecomposition
end

# INPUT
# [[1, 2]]
# [[1, 2], [2, 3]]
# [[2, 3], [3, 4]]
# [[3, 4], [4, 5]]
# [[4, 5]]

# OUTPUT
# [[1, 2]]
# [[3, 4], [4, 5]]
# [[6, 7], [7, 8]]
# [[9, 10], [10, 11]]
# [[12, 13]]

# Negative numbers correspond to Dirichlet DoFs
# in the GLOBAL space. In these examples, we
# are neglecting Dirichlet DoFs in the boundary
# of the patches (assuming they are needed)

# INPUT
# [[-1, 1]]
# [[-1, 1], [1, 2]]
# [[1, 2], [2, 3]]
# [[2, 3], [3, -2]]
# [[3, -2]]

# OUTPUT
# [[-1, 1]]
# [[-1, 2], [2, 3]]
# [[4, 5], [5, 6]]
# [[6, 7], [7, -2]]
# [[8, -2]]


# Issue: I have to pass model, reffe, and conformity, so that I can
#        build the cell_conformity instance. I would have liked to
#        avoid that, given that these were already used in order to
#        build Vh. However, I cannot extract this info out of Vh!!! :-(
function PatchFESpace(model::DiscreteModel,
                      reffe::Tuple{<:Gridap.FESpaces.ReferenceFEName,Any,Any},
                      conformity::Gridap.FESpaces.Conformity,
                      patch_decomposition::PatchDecomposition,
                      Vh::Gridap.FESpaces.SingleFieldFESpace;
                      patches_mask=Fill(false,num_patches(patch_decomposition)))

  cell_reffe = setup_cell_reffe(model,reffe)
  cell_conformity = CellConformity(cell_reffe,conformity)

  cell_dofs_ids=get_cell_dof_ids(Vh)
  num_cells_overlapped_mesh=num_cells(patch_decomposition)
  patch_cell_dofs_ids=allocate_patch_cell_dofs_ids(num_cells_overlapped_mesh,
                                                   patch_decomposition.patch_cells,
                                                   cell_dofs_ids)

  num_dofs=generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                         get_grid_topology(model),
                                         patch_decomposition.patch_cells,
                                         patch_decomposition.patch_cells_overlapped_mesh,
                                         patch_decomposition.patch_cells_faces_on_boundary,
                                         cell_dofs_ids,
                                         cell_conformity,
                                         patches_mask)

  PatchFESpace(num_dofs,patch_cell_dofs_ids,Vh,patch_decomposition)
end

Gridap.FESpaces.get_dof_value_type(a::PatchFESpace)=Gridap.FESpaces.get_dof_value_type(a.Vh)
Gridap.FESpaces.get_free_dof_ids(a::PatchFESpace)=Base.OneTo(a.num_dofs)
Gridap.FESpaces.get_cell_dof_ids(a::PatchFESpace)=a.patch_cell_dofs_ids
Gridap.FESpaces.get_cell_dof_ids(a::PatchFESpace,::Triangulation)=a.patch_cell_dofs_ids
Gridap.FESpaces.get_fe_basis(a::PatchFESpace)=get_fe_basis(a.Vh)
Gridap.FESpaces.ConstraintStyle(a::PatchFESpace)=Gridap.FESpaces.UnConstrained()
Gridap.FESpaces.get_vector_type(a::PatchFESpace)=get_vector_type(a.Vh)

function Gridap.FESpaces.scatter_free_and_dirichlet_values(f::PatchFESpace,
                                                           free_values,
                                                           dirichlet_values)
  lazy_map(Broadcasting(Gridap.Fields.PosNegReindex(free_values,dirichlet_values)),
           f.patch_cell_dofs_ids)
end

function setup_cell_reffe(model::DiscreteModel,
                          reffe::Tuple{<:Gridap.FESpaces.ReferenceFEName,Any,Any}; kwargs...)
  basis, reffe_args,reffe_kwargs = reffe
  cell_reffe = ReferenceFE(model,basis,reffe_args...;reffe_kwargs...)
end

function allocate_patch_cell_dofs_ids(num_cells_overlapped_mesh,
                                      cell_patches,
                                      cell_dof_ids)

   ptrs=Vector{Int}(undef,num_cells_overlapped_mesh+1)
   ptrs[1]=1
   cache=array_cache(cell_patches)
   cache_cdofids=array_cache(cell_dof_ids)
   gcell_overlapped_mesh=1
   for patch=1:length(cell_patches)
    cells_patch=getindex!(cache,cell_patches,patch)
    for cell in cells_patch
      current_cell_dof_ids=getindex!(cache_cdofids,cell_dof_ids,cell)
      ptrs[gcell_overlapped_mesh+1]=ptrs[gcell_overlapped_mesh]+length(current_cell_dof_ids)
      gcell_overlapped_mesh+=1
    end
   end
   #println(num_cells_overlapped_mesh, " ", gcell_overlapped_mesh)
   Gridap.Helpers.@check num_cells_overlapped_mesh+1 == gcell_overlapped_mesh
   data=Vector{Int}(undef,ptrs[end]-1)
   Gridap.Arrays.Table(data,ptrs)
end

function generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                       topology,
                                       patch_cells,
                                       patch_cells_overlapped_mesh,
                                       patch_cells_faces_on_boundary,
                                       cell_dofs_ids,
                                       cell_conformity,
                                       patches_mask)

    cache=array_cache(patch_cells)
    num_patches=length(patch_cells)
    current_dof=1
    for patch=1:num_patches
      current_patch_cells=getindex!(cache,patch_cells,patch)
      current_dof=generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                    topology,
                                    patch,
                                    current_patch_cells,
                                    patch_cells_overlapped_mesh,
                                    patch_cells_faces_on_boundary,
                                    cell_dofs_ids,
                                    cell_conformity;
                                    free_dofs_offset=current_dof,
                                    mask=patches_mask[patch])
    end
    return current_dof-1
end

# TO-THINK/STRESS:
#  1. MultiFieldFESpace case?
#  2. FESpaces which are directly defined on physical space? We think this cased is covered by
#     the fact that we are using a CellConformity instance to rely on ownership info.
# free_dofs_offset     : the ID from which we start to assign free DoF IDs upwards
# Note: we do not actually need to generate a global numbering for Dirichlet DoFs. We can
#       tag all as them with -1, as we are always imposing homogenous Dirichlet boundary
#       conditions, and thus there is no need to address the result of interpolating Dirichlet
#       Data into the FE space.
function generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                       topology,
                                       patch::Integer,
                                       patch_cells::AbstractVector{<:Integer},
                                       patch_cells_overlapped_mesh::Gridap.Arrays.Table,
                                       patch_cells_faces_on_boundary,
                                       global_space_cell_dofs_ids,
                                       cell_conformity;
                                       free_dofs_offset=1,
                                       mask=false)

  patch_global_space_cell_dofs_ids=
     lazy_map(Broadcasting(Reindex(global_space_cell_dofs_ids)),patch_cells)

  o  = patch_cells_overlapped_mesh.ptrs[patch]
  if mask
    for lpatch_cell=1:length(patch_cells)
      cell_overlapped_mesh=patch_cells_overlapped_mesh.data[o+lpatch_cell-1]
      s,e=patch_cell_dofs_ids.ptrs[cell_overlapped_mesh],
            patch_cell_dofs_ids.ptrs[cell_overlapped_mesh+1]-1
      patch_cell_dofs_ids.data[s:e] .= -1
    end
  else
    g2l=Dict{Int,Int}()
    Dc = length(patch_cells_faces_on_boundary)

    # Loop over cells of the patch (local_cell_id_within_patch)
    for (lpatch_cell,patch_cell) in enumerate(patch_cells)
      cell_overlapped_mesh=patch_cells_overlapped_mesh.data[o+lpatch_cell-1]
      s,e=patch_cell_dofs_ids.ptrs[cell_overlapped_mesh],
            patch_cell_dofs_ids.ptrs[cell_overlapped_mesh+1]-1
      current_patch_cell_dofs_ids=view(patch_cell_dofs_ids.data,s:e)
      face_offset=0
      ctype = cell_conformity.cell_ctype[patch_cell]
      for d=0:Dc-1
        cells_d_faces = Gridap.Geometry.get_faces(topology,Dc,d)
        cell_d_face   = cells_d_faces[patch_cell]
        #println(patch_cell, " ", patch_cells_faces_on_boundary[d+1][cell_overlapped_mesh])
        #println(patch_cell, " ", cell_d_face, " ", s:e)

        for (lf,f) in enumerate(cell_d_face)
          # If current face is on the patch boundary
          if (patch_cells_faces_on_boundary[d+1][cell_overlapped_mesh][lf])
            # assign negative indices to DoFs owned by face
            for ldof in cell_conformity.ctype_lface_own_ldofs[ctype][face_offset+lf]
              gdof=global_space_cell_dofs_ids[patch_cell][ldof]
              current_patch_cell_dofs_ids[ldof] = -1
              # println(ldof)
            end
          else
            # rely on the existing glued info (available at global_space_cell_dof_ids)
            # (we will need a Dict{Int,Int} to hold the correspondence among global
            # space and patch cell dofs IDs)
            for ldof in cell_conformity.ctype_lface_own_ldofs[ctype][face_offset+lf]
              gdof=global_space_cell_dofs_ids[patch_cell][ldof]
              if (gdof>0)
                if gdof in keys(g2l)
                  current_patch_cell_dofs_ids[ldof] = g2l[gdof]
                else
                  g2l[gdof] = free_dofs_offset
                  current_patch_cell_dofs_ids[ldof] = free_dofs_offset
                  free_dofs_offset += 1
                end
              else
                current_patch_cell_dofs_ids[ldof] = -1
              end
            end
          end
        end
        face_offset += cell_conformity.d_ctype_num_dfaces[d+1][ctype]
      end
      # Interior DoFs
      for ldof in cell_conformity.ctype_lface_own_ldofs[ctype][face_offset+1]
        # println("ldof: $(ldof) $(length(current_patch_cell_dofs_ids))")
        current_patch_cell_dofs_ids[ldof] = free_dofs_offset
        free_dofs_offset += 1
      end
    end
  end
  return free_dofs_offset
end


# x \in  PatchFESpace
# y \in  SingleFESpace
# TO-DO: Replace PatchFESpace by a proper operator.
function prolongate!(x::AbstractVector{T},Ph::PatchFESpace,y::AbstractVector{T}) where T
  Gridap.Helpers.@check num_free_dofs(Ph.Vh) == length(y)
  Gridap.Helpers.@check num_free_dofs(Ph) == length(x)

  # Gather y cell-wise
  y_cell_wise=scatter_free_and_dirichlet_values(Ph.Vh,
                                                y,
                                                get_dirichlet_dof_values(Ph.Vh))

  # Gather y cell-wise in overlapped mesh
  y_cell_wise_with_overlap=lazy_map(Broadcasting(Reindex(y_cell_wise)),
                                    Ph.patch_decomposition.patch_cells.data)

  Gridap.FESpaces._free_and_dirichlet_values_fill!(
    x,
    [1.0], # We need an array of size 1 as we put -1 everywhere at the patch boundaries
    array_cache(y_cell_wise_with_overlap),
    array_cache(Ph.patch_cell_dofs_ids),
    y_cell_wise_with_overlap,
    Ph.patch_cell_dofs_ids,
    Gridap.Arrays.IdentityVector(length(Ph.patch_cell_dofs_ids)))

end

# x \in  SingleFESpace
# y \in  PatchFESpace
function inject!(x,Ph::PatchFESpace,y)
  w = compute_weight_operators(Ph)
  inject!(x,Ph::PatchFESpace,y,w)
end

function inject!(x,Ph::PatchFESpace,y,w)
  touched=Dict{Int,Bool}()
  cell_mesh_overlapped=1
  cache_patch_cells=array_cache(Ph.patch_decomposition.patch_cells)
  cell_dof_ids=get_cell_dof_ids(Ph.Vh)
  cache_cell_dof_ids=array_cache(cell_dof_ids)
  fill!(x,0.0)
  for patch=1:length(Ph.patch_decomposition.patch_cells)
    current_patch_cells=getindex!(cache_patch_cells,
                                  Ph.patch_decomposition.patch_cells,
                                  patch)
    for cell in current_patch_cells
      current_cell_dof_ids=getindex!(cache_cell_dof_ids,cell_dof_ids,cell)
      s = Ph.patch_cell_dofs_ids.ptrs[cell_mesh_overlapped]
      e = Ph.patch_cell_dofs_ids.ptrs[cell_mesh_overlapped+1]-1
      current_patch_cell_dof_ids=view(Ph.patch_cell_dofs_ids.data,s:e)
      for (dof,pdof) in zip(current_cell_dof_ids,current_patch_cell_dof_ids)
        if pdof >0 && !(dof in keys(touched))
          touched[dof]=true
          x[dof]+=y[pdof]*w[pdof]
        end
      end
      cell_mesh_overlapped+=1
    end
    empty!(touched)
  end
end

function compute_weight_operators(Ph::PatchFESpace)
  cell_dof_ids=get_cell_dof_ids(Ph.Vh)
  cache_cell_dof_ids=array_cache(cell_dof_ids)
  cache_patch_cells=array_cache(Ph.patch_decomposition.patch_cells)

  w=zeros(num_free_dofs(Ph.Vh))
  touched=Dict{Int,Bool}()
  cell_mesh_overlapped=1
  for patch=1:length(Ph.patch_decomposition.patch_cells)
    current_patch_cells=getindex!(cache_patch_cells,
                                  Ph.patch_decomposition.patch_cells,
                                  patch)
    for cell in current_patch_cells
      current_cell_dof_ids=getindex!(cache_cell_dof_ids,cell_dof_ids,cell)
      s = Ph.patch_cell_dofs_ids.ptrs[cell_mesh_overlapped]
      e = Ph.patch_cell_dofs_ids.ptrs[cell_mesh_overlapped+1]-1
      current_patch_cell_dof_ids=view(Ph.patch_cell_dofs_ids.data,s:e)
      for (dof,pdof) in zip(current_cell_dof_ids,current_patch_cell_dof_ids)
        if pdof > 0 && !(dof in keys(touched))
          touched[dof]=true
          w[dof]+=1.0
        end
      end
      cell_mesh_overlapped+=1
    end
    empty!(touched)
  end
  w .= 1.0 ./ w
  w_Ph=similar(w,num_free_dofs(Ph))
  prolongate!(w_Ph,Ph,w)
  w_Ph
end
