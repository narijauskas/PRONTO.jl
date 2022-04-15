
function stage_cost!(dh, h, (l,x,u), t)
    dh .= l(x(t), u(t))
end

function cost(x,u,t,l)
    T = last(t)
    h = solve(ODEProblem(stage_cost!, [0], (0.0,T), (l,x,u)))
    return h
end
 