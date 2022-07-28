# --------------------------------- search direction forward integration --------------------------------- #
function search_z(NX,T,Ko,vo,A,B)
    z0 = zeros(NX)
    z! = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (Ko,vo,A,B)))
    Z = functor((Z,t)->z!(Z,t), buffer(NX))
    return Z
end

function update_dynamics!(dz, z, (Ko,vo,A,B), t)
    v = -Ko(t)*z+vo(t)
    dz .= A(t)*z + B(t)*v
end


function search_v(NU,z,Ko,vo)
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
