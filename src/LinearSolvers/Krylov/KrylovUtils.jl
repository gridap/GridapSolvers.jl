
"""
  Computes the Krylov matrix-vector product y = Pl⁻¹⋅A⋅Pr⁻¹⋅x
  by solving: 
    Pr⋅wr = x
    wl = A⋅wr
    Pl⋅y = wl
"""
function krylov_mul!(y,A,x,Pr,Pl,wr,wl)
  solve!(wr,Pr,x)
  mul!(wl,A,wr)
  solve!(y,Pl,wl)
end
function krylov_mul!(y,A,x,Pr,Pl::Nothing,wr,wl)
  solve!(wr,Pr,x)
  mul!(y,A,wr)
end
function krylov_mul!(y,A,x,Pr::Nothing,Pl,wr,wl)
  mul!(wl,A,x)
  solve!(y,Pl,wl)
end
function krylov_mul!(y,A,x,Pr::Nothing,Pl::Nothing,wr,wl)
  mul!(y,A,x)
end

"""
  Computes the Krylov residual r = Pl⁻¹(A⋅x - b).
  by solving: 
    w = A⋅x - b
    Pl⋅r = w
"""
function krylov_residual!(r,x,A,b,Pl,w)
  mul!(w,A,x)
  w .= b .- w
  solve!(r,Pl,w)
end
function krylov_residual!(r,x,A,b,Pl::Nothing,w)
  mul!(r,A,x)
  r .= b .- r
end
