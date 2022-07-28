# --------------------------------- projection --------------------------------- #

# for projection, provided Kr(t)
# FUTURE: in-place f!(dx,x,u) 
function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end


# η,Kr -> ξ=(x,u) # projection to generate stabilized trajectory
function projection(NX,NU,T,α,μ,Kr,f,x0)
    x! = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))
    x = buffer(NX)
    u = buffer(NU)

    # _x(t) = (x!(x, t); return x)
    function _x(t)
        x!(x, t)
        return x
    end

    function _u(t)
        mul!(u, Kr(t), α(t)-_x(t))
        u .+= μ(t)
        return u
    end

    return (_x,_u)
end

