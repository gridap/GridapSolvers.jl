
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

# MultiField/DistributedMultiField missing API

function Gridap.FESpaces.zero_dirichlet_values(f::MultiFieldFESpace)
  map(zero_dirichlet_values,f.spaces)
end

function Gridap.FESpaces.interpolate_everywhere!(objects,free_values::AbstractVector,dirichlet_values::Vector,fe::MultiFieldFESpace)
  blocks = SingleFieldFEFunction[]
  for (field, (U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    dirichlet_values_i = dirichlet_values[field]
    uhi = interpolate!(object, free_values_i, dirichlet_values_i, U)
    push!(blocks,uhi)
  end
  Gridap.MultiField.MultiFieldFEFunction(free_values,fe,blocks)
end

function Gridap.FESpaces.interpolate!(objects::GridapDistributed.DistributedMultiFieldFEFunction,free_values::AbstractVector,fe::GridapDistributed.DistributedMultiFieldFESpace)
  part_fe_fun = map(local_views(objects),partition(free_values),local_views(fe)) do objects,x,f
    interpolate!(objects,x,f)
  end
  field_fe_fun = GridapDistributed.DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(fe)
    free_values_i = Gridap.MultiField.restrict_to_field(fe,free_values,i)
    fe_space_i = fe.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i)
    push!(field_fe_fun,fe_fun_i)
  end
  GridapDistributed.DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end

function Gridap.FESpaces.FEFunction(
  f::GridapDistributed.DistributedMultiFieldFESpace,x::AbstractVector,
  dirichlet_values::AbstractArray{<:AbstractVector},isconsistent=false
  )
  free_values  = GridapDistributed.change_ghost(x,f.gids;is_consistent=isconsistent,make_consistent=true)
  part_fe_fun  = map(FEFunction,f.part_fe_space,partition(free_values))
  field_fe_fun = GridapDistributed.DistributedSingleFieldFEFunction[]
  for i in 1:num_fields(f)
    free_values_i = Gridap.MultiField.restrict_to_field(f,free_values,i)
    dirichlet_values_i = dirichlet_values[i]
    fe_space_i = f.field_fe_space[i]
    fe_fun_i = FEFunction(fe_space_i,free_values_i,dirichlet_values_i,true)
    push!(field_fe_fun,fe_fun_i)
  end
  GridapDistributed.DistributedMultiFieldFEFunction(field_fe_fun,part_fe_fun,free_values)
end
