# --------------------------------- search direction forward integration --------------------------------- #
function search_z(x,u,Ko,vo,model)
    NX = model.NX; NU = model.NU; T = model.T;
    fx! = model.fx!; _A = buffer(NX,NX)
    A = @closure (t)->(fx!(_A, x(t), u(t)); return _A)
    fu! = model.fu!; _B = buffer(NX,NU)
    B = @closure (t)->(fu!(_B, x(t), u(t)); return _B)
    # B(t) = (fu!(_B, x(t), u(t)); return _B)

    z0 = zeros(NX)
    z! = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (Ko,vo,A,B)))
    Z = functor((Z,t)->z!(Z,t), buffer(NX))
    return Z
end


function update_dynamics!(dz, z, (Ko,vo,A,B), t)
    # (Ko,vo) = Kovo(t)
    v = -Ko(t)*z+vo(t)
    dz .= A(t)*z + B(t)*v
end


function search_v(z,Ko,vo,model)
    NU = model.NU;
    V = buffer(NU)
    # v = -Ko(t)*z+vo(t)
    function _v(t)
        # (Ko,vo) = Kovo(t)
        mul!(V, Ko(t), z(t))
        V .*= -1
        V .+= vo(t)
        return V
    end
    return _v
end
