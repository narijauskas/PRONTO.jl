# --------------------------------- projection --------------------------------- #

# for projection, provided Kr(t)
# FUTURE: in-place f!(dx,x,u) 
function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end

# η,Kr -> ξ=(x,u) # projection to generate stabilized trajectory
function projection(NX,NU,T,α,μ,Kr,f,x0)
    x_ode = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))
    xbuf = Buffer(NX)
    ubuf = Buffer(NU)

    function x(t)
        copy!(xbuf, x_ode(t))
        return xbuf
    end

    function u(t)
        mul!(ubuf, Kr(t), α(t)-x(t))
        ubuf .+= μ(t)
        return ubuf
    end

    return (x,u)
end

