# --------------------------------- solve lagrangian costate dynamics --------------------------------- #

# dλ = (A-B*Kr)'λ + a - Kr'b

 function lagrange(x,u,Kr,rT,model)
    NX = model.NX; NU = model.NU; T = model.T;     
    fx! = model.fx!; _A = Buffer{Tuple{NX,NX}}()
    A = @closure (t)->(fx!(_A,x(t),u(t)); return _A)
    fu! = model.fu!; _B = Buffer{Tuple{NX,NU}}()
    B = @closure (t)->(fu!(_B,x(t),u(t)); return _B)
    lx! = model.lx!; _a = Buffer{Tuple{NX}}()
    a = @closure (t)->(lx!(_a,x(t),u(t)); return _a)
    lu! = model.lu!; _b = Buffer{Tuple{NU}}()
    b = @closure (t)->(lu!(_b,x(t),u(t)); return _b)
    luu! = model.luu!; _R = Buffer{Tuple{NU,NU}}()
    R = @closure (t)->(luu!(_R,x(t),u(t)); return _R)

    λ! = solve(ODEProblem(costate_dynamics!, collect(rT), (T,0.0), (A,B,a,b,Kr)))
    return λ!
end

# --------------------------------- costate dynamics vo --------------------------------- #
function optimizer_costate(x,u,Ko,rT,model)
    NX = model.NX; NU = model.NU; T = model.T;     
    fx! = model.fx!; _A = Buffer{Tuple{NX,NX}}()
    A = @closure (t)->(fx!(_A,x(t),u(t)); return _A)
    fu! = model.fu!; _B = Buffer{Tuple{NX,NU}}()
    B = @closure (t)->(fu!(_B,x(t),u(t)); return _B)
    lx! = model.lx!; _a = Buffer{Tuple{NX}}()
    a = @closure (t)->(lx!(_a,x(t),u(t)); return _a)
    lu! = model.lu!; _b = Buffer{Tuple{NU}}()
    b = @closure (t)->(lu!(_b,x(t),u(t)); return _b)
    luu! = model.luu!; _R = Buffer{Tuple{NU,NU}}()
    R = @closure (t)->(luu!(_R,x(t),u(t)); return _R)

    r! = solve(ODEProblem(costate_dynamics!, collect(rT), (T,0.0), (A,B,a,b,Ko)))
    _r = Buffer{Tuple{NX}}()
    r = @closure (t)->(r!(_r,t); return _r)

    _vo = Buffer{Tuple{NU}}()
    function vo(t)
        copy!(_vo, -R(t)\(B(t)'*r(t)+b(t)))
    end
    return vo
end

# function 

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + K(t)'*b(t)
end