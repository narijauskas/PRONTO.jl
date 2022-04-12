
stage_cost!(dL, L, (l,x,u), t) = dL .= l(x(t), u(t))

function cost(x,u,t,l)
    T = last(t)
    L = solve(ODEProblem(stage_cost!, [0], (0.0,T), (l,x,u)))
    return L
end
