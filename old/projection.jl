
function stabilized_dynamics!(dx, x, model, t)
    
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end


# project φ,Kr->ξ
# simulates dynamics and control law
function projection(α, μ, Kr, model)
    T = last(model.t)
    x = Timeseries( solve(ODEProblem(stabilized_dynamics!, model.x0, model.tspan, model)) )
    u = Timeseries(t->(μ(t) - Kr(t)*(x(t)-α(t))))
    # x = Timeseries(t->x(t), model.t)
    ξ = (x,u)
    return ξ
end



# projection setup

function stabilized_dynamics!(dx, x, model, t)
    # captures φ? Kr?

    μ = φ.μ(t)
    α = φ.α(t)
    Kr = Kr(model, t)

    u = μ - Kr*(x-α)
    dx .= model.f(x,u)
end