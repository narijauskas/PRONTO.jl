# --------------------------------- costate dynamics vo --------------------------------- #
function costate_dynamics(x,u,Ko,rT,model)
    NX = model.NX; NU = model.NU; T = model.T;     
    
    fx! = model.fx!;
    _A = Buffer{Tuple{NX,NX}}()
    A(t) = (fx!(_A,x(t),u(t)); return _A)
    
    fu! = model.fu!;
    _B = Buffer{Tuple{NX,NU}}()
    B(t) = (fu!(_B,x(t),u(t)); return _B)
    
    lx! = model.lx!;
    _a = Buffer{Tuple{NX}}()
    a(t) = (lx!(_a,x(t),u(t)); return _a)
    
    lu! = model.lu!;
    _b = Buffer{Tuple{NU}}()
    b(t) = (lu!(_b,x(t),u(t)); return _b)
    
    luu! = model.luu!;
    _R = Buffer{Tuple{NU,NU}}()
    R(t) = (luu!(_R,x(t),u(t)); return _R)

    r! = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)))
    r = functor((_r,t)->r!(_r,t), Buffer{Tuple{NX}}())

    _vo = Buffer{Tuple{NU}}()
    #TODO: vo(x,u)
    function vo(t)
        copy!(_vo, -R(t)\(B(t)'*r(t)+b(t)))
    end
    return vo
end

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + K(t)'*b(t)
end