

function Gridap.Adaptivity.change_domain_n2o(f_fine,ctrian::Gridap.Geometry.Triangulation{Dc},glue::Gridap.Adaptivity.AdaptivityGlue{<:Gridap.Adaptivity.RefinementGlue,Dc}) where Dc
  @notimplementedif num_dims(ctrian) != Dc
  msg = "Evaluating a fine CellField in the coarse mesh is costly! If you are using this feature 
         to integrate, consider using a CompositeMeasure instead (see test/AdaptivityTests/GridTransferTests.jl)."
  @warn msg

  if (num_cells(ctrian) != 0)
    # f_c2f[i_coarse] = [f_fine[i_fine_1], ..., f_fine[i_fine_nChildren]]
    f_c2f = Gridap.Adaptivity.f2c_reindex(f_fine,glue)

    child_ids = Gridap.Adaptivity.f2c_reindex(glue.n2o_cell_to_child_id,glue)
    rrules    = Gridap.Adaptivity.get_old_cell_refinement_rules(glue)
    f_coarse  = lazy_map(Gridap.Adaptivity.FineToCoarseField,f_c2f,rrules,child_ids)
    return Gridap.CellData.GenericCellField(f_coarse,ctrian,ReferenceDomain())
  else
    f_coarse = Fill(Gridap.Fields.ConstantField(0.0),num_cells(ftrian))
    return Gridap.CellData.GenericCellField(f_coarse,ctrian,ReferenceDomain())
  end
end

function Gridap.Adaptivity.FineToCoarseField(fine_fields::AbstractArray{<:Gridap.Fields.Field},rrule::Gridap.Adaptivity.RefinementRule,child_ids::AbstractArray{<:Integer})
  fields = Vector{Gridap.Fields.Field}(undef,Gridap.Adaptivity.num_subcells(rrule))
  fields = fill!(fields,Gridap.Fields.ConstantField(0.0))
  for (k,id) in enumerate(child_ids)
    fields[id] = fine_fields[k]
  end
  return Gridap.Adaptivity.FineToCoarseField(fields,rrule)
end


function Base.map(::typeof(Gridap.Arrays.testitem),
  a::Tuple{<:AbstractVector{<:AbstractVector{<:VectorValue}},<:AbstractVector{<:Gridap.Fields.LinearCombinationFieldVector}})
  a2=Gridap.Arrays.testitem(a[2])
  a1=Vector{eltype(eltype(a[1]))}(undef,size(a2,1))
  a1.=zero(Gridap.Arrays.testitem(a1))
  (a1,a2)
end

# This fix is required to be able to integrate in the overlapped mesh underlying patch smoothers
function Gridap.Geometry.get_glue(trian::BodyFittedTriangulation{Dt},::Val{Dt}) where Dt
  tface_to_mface = trian.tface_to_mface
  tface_to_mface_map = FillArrays.Fill(Gridap.Fields.GenericField(identity),num_cells(trian))
  if isa(tface_to_mface,Gridap.Arrays.IdentityVector) && num_faces(trian.model,Dt) == num_cells(trian)
    mface_to_tface = tface_to_mface
  else
    #nmfaces = num_faces(trian.model,Dt)
    # Crashes here!!! It does not support overlapping!!!
    mface_to_tface = nothing #PosNegPartition(tface_to_mface,Int32(nmfaces))
  end
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
end
