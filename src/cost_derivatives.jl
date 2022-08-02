# --------------------------------- cost derivatives --------------------------------- #

function cost_derivatives(x,u,z,v,rT,PT,model)
    NX = model.NX; NU = model.NU; T = model.T;
    
    lx! = model.lx!;
    _a = Buffer{Tuple{NX}}()
    a(t) = (lx!(_a,x(t),u(t)); return _a)
    
    lu! = model.lu!;
    _b = Buffer{Tuple{NU}}()
    b(t) = (lu!(_b,x(t),u(t)); return _b)
    
    lxx! = model.lxx!;
    _Q = Buffer{Tuple{NX,NX}}()
    Q(t) = (lxx!(_Q,x(t),u(t)); return _Q)
    
    luu! = model.luu!;
    _R = Buffer{Tuple{NU,NU}}()
    R(t) = (luu!(_R,x(t),u(t)); return _R)
    
    lxu! = model.lxu!;
    _S = Buffer{Tuple{NX,NU}}()
    S(t) = (lxu!(_S,x(t),u(t)); return _S)
    
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
