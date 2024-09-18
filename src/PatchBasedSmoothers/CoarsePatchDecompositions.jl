
# PatchDecomposition where each patch is given by a coarse cell on the parent mesh
function CoarsePatchDecomposition(
  model::Gridap.Adaptivity.AdaptedDiscreteModel{Dc,Dp},
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude(),
  boundary_tag_names::AbstractArray{String}=["boundary"]
) where {Dc,Dp}
  glue = Gridap.Adaptivity.get_adaptivity_glue(model)

  patch_cells = glue.o2n_faces_map
  patch_facets = get_coarse_patch_facets(model, patch_cells)
  patch_cells_faces_on_boundary = compute_patch_cells_faces_on_boundary(
    model, patch_cells, patch_facets, patch_boundary_style, boundary_tag_names
  )

  return PatchDecomposition{Dc,Dc,Dp}(
    model, patch_cells, patch_cells_faces_on_boundary, patch_boundary_style
  )
end

function get_coarse_patch_facets(
  model::Gridap.Adaptivity.AdaptedDiscreteModel{Dc,Dp},patch_cells
) where {Dc,Dp}
  topo = get_grid_topology(model)
  c2f_map = Geometry.get_faces(topo,Dc,Dc-1)
  f2c_map = Geometry.get_faces(topo,Dc-1,Dc)
  
  touched = fill(false,num_faces(topo,Dc-1))
  ptrs = fill(0,length(patch_cells)+1)
  for (patch,cells) in enumerate(patch_cells)
    for c in cells
      for f in c2f_map[c]
        nbors = f2c_map[f]
        if !touched[f] && (length(nbors) == 2) && all(n -> n ∈ cells, nbors)
          touched[f] = true
          ptrs[patch+1] += 1
        end
      end
    end
  end
  Arrays.length_to_ptrs!(ptrs)
  
  data = fill(0,ptrs[end]-1)
  fill!(touched,false)
  for (patch,cells) in enumerate(patch_cells)
    for c in cells
      for f in c2f_map[c]
        nbors = f2c_map[f]
        if !touched[f] && (length(nbors) == 2) && all(n -> n ∈ cells, nbors)
          touched[f] = true
          data[ptrs[patch]] = f
          ptrs[patch] += 1
        end
      end
    end
  end
  Arrays.rewind_ptrs!(ptrs)
  
  patch_facets = Table(data,ptrs)
  return patch_facets
end
