# --------------------------------- search direction forward integration --------------------------------- #
function search_z(x,u,Ko,vo,model)
    NX = model.NX; NU = model.NU; T = model.T;
    
    fx! = model.fx!;
    _A = Buffer{Tuple{NX,NX}}()
    A(t) = (fx!(_A, x(t), u(t)); return _A)
    
    fu! = model.fu!;
    _B = Buffer{Tuple{NX,NU}}()
    B(t) = (fu!(_B, x(t), u(t)); return _B)

    z0 = zeros(NX)
    z! = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (Ko,vo,A,B)))
    z = functor((_z,t)->z!(_z,t), Buffer{Tuple{NX}}())
    return z
end


function update_dynamics!(dz, z, (Ko,vo,A,B), t)
    v = -Ko(t)*z+vo(t)
    dz .= A(t)*z + B(t)*v
end


function search_v(z,Ko,vo,model)
    NU = model.NU;
    _v = buffer(NU)
    # v = -Ko(t)*z+vo(t)
    function v(t)
        mul!(_v, Ko(t), z(t))
        _v .*= -1
        _v .+= vo(t)
        return _v
    end
    return v
end
