
# DistributedRefinedDiscreteModels

const DistributedRefinedDiscreteModel{Dc,Dp} = GridapDistributed.DistributedDiscreteModel{Dc,Dp,<:AbstractPData{<:RefinedDiscreteModel{Dc,Dp}}}

function DistributedRefinedDiscreteModel(model::GridapDistributed.AbstractDistributedDiscreteModel,
                                         parent_models::AbstractPData{<:DiscreteModel},
                                         glue::AbstractPData{<:RefinementGlue})
                                         
  models = map_parts(local_views(model),parent_models,glue) do model, parent, glue
    RefinedDiscreteModel(model,parent,glue)
  end
  return GridapDistributed.DistributedDiscreteModel(models,get_cell_gids(model))
end

function DistributedRefinedDiscreteModel(model::GridapDistributed.AbstractDistributedDiscreteModel,
                                         parent::GridapDistributed.AbstractDistributedDiscreteModel,
                                         glue::AbstractPData{<:RefinementGlue})
  if !(model.parts === parent.parts)
    parent_models = map_parts(local_views(model)) do m
      if i_am_in(parent.parts)
        parent.dmodel.models.part
      else
        nothing
      end
    end
    return DistributedRefinedDiscreteModel(model,parent_models,glue)
  else
    return DistributedRefinedDiscreteModel(model,local_views(parent),glue)
  end
end

# DistributedRefinedTriangulations

const DistributedRefinedTriangulation{Dc,Dp} = GridapDistributed.DistributedTriangulation{Dc,Dp,<:AbstractPData{<:RefinedTriangulation{Dc,Dp}}}


# DistributedFESpaces

function get_test_space(U::GridapDistributed.DistributedSingleFieldFESpace)
  spaces = map_parts(local_views(U)) do U
    U.space
  end
  gids = U.gids
  vector_type = U.vector_type
  return GridapDistributed.DistributedSingleFieldFESpace(spaces,gids,vector_type)
end

function FESpaces.get_triangulation(f::GridapDistributed.DistributedSingleFieldFESpace,model::GridapDistributed.AbstractDistributedDiscreteModel)
  trians = map_parts(get_triangulation,local_views(f))
  GridapDistributed.DistributedTriangulation(trians,model)
end


# Refinement Operators

function Gridap.Refinement.ProjectionTransferOperator(from::GridapDistributed.DistributedFESpace,
                                                      Ω_from::GridapDistributed.DistributedTriangulation,
                                                      to::GridapDistributed.DistributedFESpace,
                                                      Ω_to::GridapDistributed.DistributedTriangulation;
                                                      solver::LinearSolver=BackslashSolver(),
                                                      Π=Gridap.Refinement.Π_l2, 
                                                      qdegree=3)
  #@assert is_change_possible(Ω_from,Ω_to)

  # Choose integration space
  Ω  = best_target(Ω_from,Ω_to)
  dΩ = Measure(Ω,qdegree)
  U  = (Ω === Ω_from) ? from : to
  V  = get_test_space(U)
  vh_to = get_fe_basis(to)
  #vh = change_domain(vh_to,Ω)
  vh  = (Ω === Ω_from) ? change_domain_c2f(vh_to,Ω,Ω.model.glue) : vh_to

  # Prepare system
  V_to = get_test_space(to)
  lhs_mat, lhs_vec = assemble_lhs(Π,Ω_to,to,V_to,qdegree)
  rhs_vec = similar(lhs_vec)
  assem   = SparseMatrixAssembler(to,V_to)

  # Prepare solver
  ss = symbolic_setup(solver,lhs_mat)
  ns = numerical_setup(ss,lhs_mat)

  caches = ns, lhs_vec, rhs_vec, Π, assem, Ω, dΩ, U, V, vh, Ω_to
  return Gridap.Refinement.ProjectionTransferOperator(eltype(sysmat),from,to,caches)
end

# Solves the problem Π(uh,vh)_to = Π(uh_from,vh)_Ω for all vh in Vh_to
function LinearAlgebra.mul!(y::PVector,A::Gridap.Refinement.ProjectionTransferOperator,x::PVector)
  ns, lhs_vec, rhs_vec, Π, assem, Ω, dΩ, U, V, vh_Ω, Ω_to = A.caches

  # Bring uh to the integration domain
  uh_from = FEFunction(A.from,x)
  uh_Ω    = change_domain(uh_from,Ω,ReferenceDomain())

  # Assemble rhs vector
  contr   = Π(uh_Ω,vh_Ω,dΩ)
  if Ω !== Ω_to
    contr = merge_contr_cells(contr,Ω,Ω_to)
  end
  vecdata = collect_cell_vector(A.to.space,contr)
  assemble_vector!(rhs_vec,assem,vecdata)
  rhs_vec .-= lhs_vec

  # Solve projection
  solve!(y,ns,sysvec)
  return y
end


# ChangeDomain

function Gridap.Refinement.merge_contr_cells(a::GridapDistributed.DistributedDomainContribution,
                                             rtrian::GridapDistributed.DistributedTriangulation,
                                             ctrian::GridapDistributed.DistributedTriangulation)
  b = map_parts(Gridap.Refinement.merge_contr_cells,local_views(a),local_views(rtrian),local_views(ctrian))
  return GridapDistributed.DistributedDomainContribution(b)
end

function Gridap.Refinement.change_domain_c2f(c_cell_field,
  ftrian::GridapDistributed.DistributedTriangulation{Dc,Dp},
  glue::MPIData{<:Union{Nothing,Gridap.Refinement.RefinementGlue}}) where {Dc,Dp}

  i_am_in_coarse = (c_cell_field != nothing)

  fields = map_parts(local_views(ftrian)) do Ω
    if (i_am_in_coarse)
      c_cell_field.fields.part
    else
      Gridap.Helpers.@check num_cells(Ω) == 0
      Gridap.CellData.GenericCellField(Fill(Gridap.Fields.ConstantField(0.0),num_cells(Ω)),Ω,ReferenceDomain())
    end
  end
  c_cell_field_fine = GridapDistributed.DistributedCellField(fields)

  dfield = map_parts(Gridap.Refinement.change_domain_c2f,local_views(c_cell_field_fine),local_views(ftrian),glue)
  return GridapDistributed.DistributedCellField(dfield)
end
