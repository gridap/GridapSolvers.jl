
function Gridap.Adaptivity.change_domain_n2o(f_fine,ftrian::Gridap.Adaptivity.AdaptedTriangulation{Dc},ctrian::Gridap.Geometry.Triangulation,glue::Gridap.Adaptivity.AdaptivityGlue{<:Gridap.Adaptivity.RefinementGlue}) where Dc
  fglue = Gridap.Geometry.get_glue(ftrian,Val(Dc))
  cglue = Gridap.Geometry.get_glue(ctrian,Val(Dc))

  @notimplementedif Gridap.Geometry.num_point_dims(ftrian) != Dc
  @notimplementedif isa(cglue,Nothing)

  if (num_cells(ctrian) != 0)
    ### New Triangulation -> New Model
    fine_tface_to_field = Gridap.CellData.get_data(f_fine)
    fine_mface_to_field = Gridap.Geometry.extend(fine_tface_to_field,fglue.mface_to_tface)

    ### New Model -> Old Model
    # f_c2f[i_coarse] = [f_fine[i_fine_1], ..., f_fine[i_fine_nChildren]]
    f_c2f = Gridap.Adaptivity.f2c_reindex(fine_mface_to_field,glue)

    child_ids = Gridap.Adaptivity.f2c_reindex(glue.n2o_cell_to_child_id,glue)
    rrules    = Gridap.Adaptivity.get_old_cell_refinement_rules(glue)
    coarse_mface_to_field = lazy_map(Gridap.Adaptivity.FineToCoarseField,f_c2f,rrules,child_ids)

    ### Old Model -> Old Triangulation
    coarse_tface_to_field = lazy_map(Reindex(coarse_mface_to_field),cglue.tface_to_mface)
    f_coarse = lazy_map(Broadcasting(∘),coarse_tface_to_field,cglue.tface_to_mface_map)

    return Gridap.CellData.similar_cell_field(f_fine,f_coarse,ctrian,ReferenceDomain())
  else
    f_coarse = Fill(Gridap.Fields.ConstantField(0.0),num_cells(fcoarse))
    return Gridap.CellData.similar_cell_field(f_fine,f_coarse,ctrian,ReferenceDomain())
  end
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
