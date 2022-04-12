

function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= A(t)'*P+P*A(t)-Ko'*R(t)*Ko+Q(t)
end





function gradient_descent(x,u,t,model)
    A = t->model[:fx](x(t), u(t))
    B = t->model[:fu](x(t), u(t))
    a = t->model[:lx](x(t), u(t))'
    b = t->model[:lu](x(t), u(t))'
    Q = t->model[:lxx](x(t), u(t))
    R = t->model[:luu](x(t), u(t))
    S = t->model[:lxu](x(t), u(t))
    T = last(t)
    Pt = model[:pxx](x(T))

    P = solve(ODEProblem(optimizer!, Pt, (T,0.0), (A,B,Q,R,S)), Rosenbrock23())
    Ko = tau(τ->(R(τ)\(S(τ)'+B(τ)'*P(τ)))', t)
    return Ko
end