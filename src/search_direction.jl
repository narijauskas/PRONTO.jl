# --------------------------------- search direction forward integration --------------------------------- #
function search_z(x,u,Ko,vo,model)
    NX = model.NX; NU = model.NU; T = model.T;
    fx! = model.fx!; fu! = model.fu!;

    A = functor(@closure((A,t) -> fx!(A,x(t),u(t))), buffer(NX,NX))
    B = functor(@closure((B,t) -> fu!(B,x(t),u(t))), buffer(NX,NU))

    z0 = zeros(NX)
    z! = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (Ko,vo,A,B)))
    Z = functor((Z,t)->z!(Z,t), buffer(NX))
    return Z
end

function update_dynamics!(dz, z, (Ko,vo,A,B), t)
    v = -Ko(t)*z+vo(t)
    dz .= A(t)*z + B(t)*v
end


function search_v(z,Ko,vo,model)
    NU = model.NU;
    V = buffer(NU)
    # v = -Ko(t)*z+vo(t)
    function _v(t)
        mul!(V, Ko(t), z(t))
        V .*= -1
        V .+= vo(t)
        return V
    end
    return _v
end
