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
function projection(α, μ, t, Kr, x0, f, T)
    # solve ode problem for x
    sln = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (μ,α,Kr,f)))
    # remap u
    u = LinearInterpolation(hcat(map(τ->μ(τ) - Kr(τ)*(sln(τ)-α(τ)), t)...), t)
    x = LinearInterpolation(hcat(map(τ->sln(τ), t)...), t)
    return x,u
end



# cost
# simulates dL = l(x,u)
# lpcost(l,p,x,u)
# cost = L(T) + p(x(T))