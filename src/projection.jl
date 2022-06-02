
function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end

# simulates dynamics and control law
function projection(α, μ, Kr, model)
    T = last(model.t)
    x = Timeseries( solve(ODEProblem(stabilized_dynamics!, model.x0, (0.0,T), (α,μ,Kr,model.f))) )
    u = Timeseries(t->(μ(t) - Kr(t)*(x(t)-α(t))))
    # x = Timeseries(t->x(t), model.t)
    ξ = (x,u)
    return ξ
end
