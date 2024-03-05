
struct LocalProjectionMap{A,B,C} <: Map
  reffe :: A
  quad  :: B
  Mq    :: C
end

# Constructors 

function LocalProjectionMap(reffe::ReferenceFE,quad::Quadrature)
  q = get_shapefuns(reffe)
  pq = get_coordinates(quad)
  wq = get_weights(quad)

  aq = Fields.BroadcastOpFieldArray(⋅,q,transpose(q))
  Mq = evaluate(IntegrationMap(),evaluate(aq,pq),wq)
  Mq_factorized = cholesky(Mq)
  return LocalProjectionMap(reffe,quad,Mq_factorized)
end

function LocalProjectionMap(reffe::ReferenceFE,qorder)
  quad = Quadrature(get_polytope(reffe),qorder)
  return LocalProjectionMap(reffe,quad)
end

function LocalProjectionMap(poly::Polytope,name::ReferenceFEName,args...;quad_order=-1,kwargs...)
  reffe = ReferenceFE(poly,name,args...;kwargs...)
  if quad_order == -1
    quad_order = 2*(get_order(reffe)+1)
  end
  return LocalProjectionMap(reffe,quad_order)
end

# Action on Field / Array{<:Field}

function Arrays.return_cache(k::LocalProjectionMap,f)
  q = get_shapefuns(k.reffe)
  pq = get_coordinates(k.quad)
  wq = get_weights(k.quad)

  lq = Fields.BroadcastOpFieldArray(⋅,q,transpose(f))
  eval_cache = return_cache(lq,pq)
  lqx = evaluate!(eval_cache,lq,pq)
  integration_cache = return_cache(IntegrationMap(),lqx,wq)
  return eval_cache, integration_cache
end

function Arrays.evaluate!(cache,k::LocalProjectionMap,f)
  eval_cache, integration_cache = cache
  q = get_shapefuns(k.reffe)

  lq = Fields.BroadcastOpFieldArray(⋅,q,transpose(f))
  lqx = evaluate!(eval_cache,lq,get_coordinates(k.quad))
  bq = evaluate!(integration_cache,IntegrationMap(),lqx,get_weights(k.quad))

  λ = ldiv!(k.Mq,bq)
  return linear_combination(λ,q)
end

# Action on CellField / DistributedCellField

function (k::LocalProjectionMap)(f::CellField)
  @assert isa(DomainStyle(f),ReferenceDomain)
  f_data = CellData.get_data(f)
  fk_data = lazy_map(k,f_data)
  return GenericCellField(fk_data,get_triangulation(f),ReferenceDomain())
end

function (k::LocalProjectionMap)(f::GridapDistributed.DistributedCellField)
  fields = map(k,local_views(f))
  return GridapDistributed.DistributedCellField(fields)
end
