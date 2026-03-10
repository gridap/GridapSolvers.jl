
using Gridap
using GridapDistributed
using PartitionedArrays
using GridapSolvers
using MPI

using Gridap.Arrays, Gridap.Geometry, Gridap.FESpaces
using GridapSolvers.MultilevelTools

using GridapSolvers.MultilevelTools: RedistributionOperator
using GridapSolvers.MultilevelTools: redistribution_cache

print_item(r, x) = println("   > Rank $r: $x")
function print_item(r, x::AbstractLocalIndices)
  println("   > Rank $r: ")
  println("      > no: $(own_length(x))")
  println("      > ng: $(ghost_length(x))")
  println("      > l2g: $(local_to_global(x))")
  println("      > l2o: $(local_to_owner(x))")
end

function mpi_print(x, name="")
  if !isnothing(x)
    ranks = linear_indices(x)
    nr = length(ranks)
    map(ranks, x) do r, x
      isa(x,DebugArray) || sleep(0.5*(r-1))
      if r == 1
        println("-------------------------------------------")
        println(" > $name : ")
      end
      print_item(r,x)
      if r == nr
        println("-------------------------------------------")
      end
    end
    PartitionedArrays.barrier(x)
  end
end

function generate_matrix_gids(mh, sh)
  lhs(u::GridapDistributed.DistributedCellField,v,dΩ) = ∫(u*v)*dΩ
  rhs(v::GridapDistributed.DistributedCellField,dΩ) = ∫(v)*dΩ
  lhs((u1,u2),(v1,v2),dΩ) = ∫(u1*v1 + u2*v2)*dΩ
  rhs((v1,v2),dΩ) = ∫(v1 + v2)*dΩ

  meas_red = Measure(Triangulation(get_model(mh)),6)
  space_red = get_fe_space(sh)
  rhs_red(v) = rhs(v,meas_red)
  lhs_red(u,v) = lhs(u,v,meas_red)
  op_red = AffineFEOperator(lhs_red, rhs_red, space_red, space_red)
  ids_red = axes(get_matrix(op_red), 2)

  meas = Measure(Triangulation(get_model_before_redist(mh)),6)
  space = get_fe_space_before_redist(sh)
  rhs(v) = rhs(v,meas)
  lhs(u,v) = lhs(u,v,meas)
  op = AffineFEOperator(lhs, rhs, space, space)
  ids = axes(get_matrix(op), 2)

  return partition(ids_red), partition(ids)
end

D = 3
ranks = DebugArray(LinearIndices((4,)))

if D == 2
  mh = CartesianModelHierarchy(ranks,[(2,2),(1,1)],(0,1,0,1),(4,4))
else
  mh = CartesianModelHierarchy(ranks,[(2,2,1),(1,1,1)],(0,1,0,1,0,1),(2,2,2))
end

# sh1 = FESpace(mh, ReferenceFE(lagrangian, Float64, 1))[1];
# sh2 = FESpace(mh, ReferenceFE(lagrangian, Float64, 2))[1];
# sh = MultiFieldFESpace([sh1, sh2]);

order = 1
sh1 = FESpace(mh, ReferenceFE(lagrangian,VectorValue{D,Float64},order); dirichlet_tags="boundary")[1];
sh2 = FESpace(mh, ReferenceFE(raviart_thomas,Float64,order-1); dirichlet_tags="boundary")[1];
sh = MultiFieldFESpace([sh1, sh2]);

reverse = false

op1 = RedistributionOperator(sh1,reverse);
x1 = zero_free_values(get_fe_space_before_redist(sh1))
y1 = zero_free_values(get_fe_space(sh1))
redistribute(y1, op1, x1)

op2 = RedistributionOperator(sh2,reverse);
x2 = zero_free_values(get_fe_space_before_redist(sh2))
y2 = zero_free_values(get_fe_space(sh2))
redistribute(y2, op2, x2)

op = RedistributionOperator(sh,reverse);
x = zero_free_values(get_fe_space_before_redist(sh))
y = zero_free_values(get_fe_space(sh))
redistribute(y, op, x)

#########################################################################################


ids_red_1, ids_1 = generate_matrix_gids(mh[1], sh1);
x1 = pzeros(ids_1)
y1 = pzeros(ids_red_1)
redistribute(y1, op1, x1)

ids_red_2, ids_2 = generate_matrix_gids(mh[1], sh2);
x2 = pzeros(ids_2)
y2 = pzeros(ids_red_2)
redistribute(y2, op2, x2)

ids_red, ids = generate_matrix_gids(mh[1], sh);
x = pzeros(ids)
y = pzeros(ids_red)
redistribute(y, op, x)

#####################################################################################

c1 = redistribution_cache(y1,x1,op1.indices_to, op1.indices_from)

(values_to, values_from), (x_ids_to, x_ids_from), (x_to, x_from), caches = c1

x_ids_to
op1.indices_to

x_ids_from
op1.indices_from