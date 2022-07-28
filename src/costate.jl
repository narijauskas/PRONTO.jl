# --------------------------------- costate dynamics vo --------------------------------- #
function costate_dynamics(NX,NU,T,x,u,α,Ko,fx!,fu!,lx!,lu!,luu!,px!)
    A = functor((A,t) -> fx!(A,x(t),u(t)), buffer(NX,NX))
    B = functor((B,t) -> fu!(B,x(t),u(t)), buffer(NX,NU))
    a = functor((a,t) -> lx!(a,x(t),u(t)), buffer(NX))
    b = functor((b,t) -> lu!(b,x(t),u(t)), buffer(NU))
    R = functor((R,t) -> luu!(R,x(t),u(t)), buffer(NU,NU))

    
    rT = buffer(NX); px!(rT, α(T)) # around unregulated trajectory
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