# inputs:
# Kr(t)
# X(t), U(t) (as estimate: α,μ)

# model elements:
# f, l, x0

# outputs:
# X(t), U(t) (as stabilized trajectory)
# L(t) (cost)

function stabilized_dynamics!(dx, x, (μ,α,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end

# simulates dynamics and control law
function projection(α, μ, Kr, model)
    T = last(model.t)
    # solve ode problem for x
    x = solve(ODEProblem(stabilized_dynamics!, model.x0, (0.0,T), (μ,α,Kr,model.f)), dt=0.001)
    u = Trajectory(t->μ(t) - Kr(t)*(x(t)-α(t)), model.t)
    x = Trajectory(t->x(t), model.t)
    return x,u
end



# cost
# simulates dL = l(x,u)
# lpcost(l,p,x,u)
# cost = L(T) + p(x(T))