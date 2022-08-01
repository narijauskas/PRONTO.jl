# --------------------------------- cost derivatives --------------------------------- #

function cost_derivatives(x,u,z,v,rT,PT,model)
    NX = model.NX; NU = model.NU; T = model.T;
    lx! = model.lx!; lu! = model.lu!;
    lxx! = model.lxx!; luu! = model.luu!; lxu! = model.lxu!;

    a = functor(@closure((a,t) -> lx!(a,x(t),u(t))), buffer(NX))
    b = functor(@closure((b,t) -> lu!(b,x(t),u(t))), buffer(NU))
    Q = functor(@closure((Q,t) -> lxx!(Q,x(t),u(t))), buffer(NX,NX))
    R = functor(@closure((R,t) -> luu!(R,x(t),u(t))), buffer(NU,NU))
    S = functor(@closure((S,t) -> lxu!(S,x(t),u(t))), buffer(NX,NU))

    y0 = [0;0]
    y = solve(ODEProblem(cost_derivatives!, y0, (0.0,T), (z,v,a,b,Q,S,R)))
    Dh = y(T)[1] + rT'*z(T)
    D2g = y(T)[2] + z(T)'*PT*z(T)
    return (Dh,D2g)
end

function cost_derivatives!(dy, y, (z,v,a,b,Qo,So,Ro), t)
    dy[1] = a(t)'*z(t) + b(t)'*v(t)
    dy[2] = z(t)'*Qo(t)*z(t) + 2*z(t)'*So(t)*v(t) + v(t)'*Ro(t)*v(t)
end
