# --------------------------------- costate dynamics vo --------------------------------- #
function costate_dynamics(Ko,A,B,a,b,R,rT,NX,NU,T)
    r! = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)))
    r = functor((r,t)->r!(r,t), buffer(NX))

    vo = buffer(NU)
    function _vo(t)
        copy!(vo, -R(t)\(B(t)'*r(t)+b(t)))
    end
    return _vo
end

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + K(t)'*b(t)
end