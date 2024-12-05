
struct PatchClosureTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}
  trian :: PatchTriangulation{Dc,Dp}
end

function Closure(PD::PatchDecomposition)
  patch_cells = generate_patch_closures(PD)
  trian  = Triangulation(PD.model)
  ptrian = PatchTriangulation(trian,PD,patch_cells,nothing)
  return PatchClosureTriangulation(ptrian)
end

function Geometry.get_background_model(t::PatchClosureTriangulation)
  get_background_model(t.trian)
end

function Geometry.get_grid(t::PatchClosureTriangulation)
  get_grid(t.trian)
end

function Geometry.get_glue(t::PatchClosureTriangulation,::Val{d}) where d
  get_glue(t.trian,Val(d))
end

function Geometry.get_facet_normal(trian::PatchClosureTriangulation)
  get_facet_normal(trian.trian)
end

function Geometry.is_change_possible(strian::PatchClosureTriangulation,ttrian::PatchClosureTriangulation)
  return is_change_possible(strian.trian,ttrian.trian)
end

function Geometry.is_change_possible(strian::PatchTriangulation,ttrian::PatchClosureTriangulation)
  return is_change_possible(strian,ttrian.trian)
end

function Geometry.move_contributions(scell_to_val::AbstractArray,strian::PatchClosureTriangulation)
  vals, _ = move_contributions(scell_to_val,strian.trian)
  return vals, strian
end
