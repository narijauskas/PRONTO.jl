# --------------------------------- costate dynamics vo --------------------------------- #
function costate_dynamics(x,u,Ko,rT,model)
    NX = model.NX; NU = model.NU; T = model.T;     
    fx! = model.fx!; _A = buffer(NX,NX)
    A = @closure (t)->(fx!(_A,x(t),u(t)); return _A)
    fu! = model.fu!; _B = buffer(NX,NU)
    B = @closure (t)->(fu!(_B,x(t),u(t)); return _B)
    lx! = model.lx!; _a = buffer(NX)
    a = @closure (t)->(lx!(_a,x(t),u(t)); return _a)
    lu! = model.lu!; _b = buffer(NU)
    b = @closure (t)->(lu!(_b,x(t),u(t)); return _b)
    luu! = model.luu!; _R = buffer(NU,NU)
    R = @closure (t)->(luu!(_R,x(t),u(t)); return _R)

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