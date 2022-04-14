

function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + collect(K(t)'*b(t))
end


function gradient_descent(x,u,t,model,Kr,x_eq)
    A = t->Main.fx(x(t), u(t))
    B = t->Main.fu(x(t), u(t))
    a = t->Main.l_x(x(t), u(t))
    b = t->Main.l_u(x(t), u(t))
    Q = t->Main.lxx(x(t), u(t))
    R = t->Main.luu(x(t), u(t))
    S = t->Main.lxu(x(t), u(t))
    T = last(t)
    PT = model[:pxx](x_eq) # use x_eq

    P = solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S)), dt=1e-3)
    Ko = t->inv(R(t))*(S(t)'+B(t)'*P(t))

    rT = model[:px](x_eq) # use x_eq
    r = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)), Rosenbrock23(), dt=0.001)
    vo = tau(τ->(-R(τ)\(B(τ)'*r(τ)+b(τ))), t)

    qT = rT # use x_eq
    q = solve(ODEProblem(costate_dynamics!, qT, (T,0.0), (A,B,a,b,Kr)), dt=0.001)
    # q = tau(τ->q(τ), t)
    # return Ko,P
    return Ko,vo,q
end
