
function stage_cost!(dh, h, (l,x,u), t)
    dh .= l(x(t), u(t))
end

function cost(x,u,model)
    T = last(model.t)
    h = solve(ODEProblem(stage_cost!, [0], (0.0,T), (model.l,x,u)))
    return h
end
 