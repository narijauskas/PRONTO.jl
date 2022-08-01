# --------------------------------- costate dynamics vo --------------------------------- #
function costate_dynamics(x,u,Ko,rT,model)
    fx! = model.fx!; fu! = model.fu!;
    lx! = model.lx!; lu! = model.lu!;
    luu! = model.luu!;
    NX = model.NX; NU = model.NU; T = model.T;     

    A = functor(@closure((A,t) -> fx!(A,x(t),u(t))), buffer(NX,NX))
    B = functor(@closure((B,t) -> fu!(B,x(t),u(t))), buffer(NX,NU))
    a = functor(@closure((a,t) -> lx!(a,x(t),u(t))), buffer(NX))
    b = functor(@closure((b,t) -> lu!(b,x(t),u(t))), buffer(NU))
    # Q = functor(@closure((Q,t) -> lxx!(Q,x(t),u(t))), buffer(NX,NX))
    R = functor(@closure((R,t) -> luu!(R,x(t),u(t))), buffer(NU,NU))
    # S = functor(@closure((S,t) -> lxu!(S,x(t),u(t))), buffer(NX,NU))

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