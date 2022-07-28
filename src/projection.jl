# --------------------------------- projection --------------------------------- #

function projection_x(NX,T,α,μ,Kr,f,x0)
    x! = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))
    X = functor((X,t)->x!(X,t), buffer(NX))
    return X
end


function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end


function projection_u(NX,NU,α,μ,Kr,x)
    U = buffer(NU)
    X = buffer(NX)
    function _u(t)
        # u = μ - Kr*(x-α)
        copy!(X, α(t))
        X .-= x(t)
        mul!(U, Kr(t), X)
        U .+= μ(t)
        return U
    end
    return _u
end





# --------------------------------- trajectories --------------------------------- #
#=
# η,Kr -> ξ=(x,u) # projection to generate stabilized trajectory
function projection(NX,NU,T,α,μ,Kr,f,x0)
    x! = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))
    X = buffer(NX)
    U = buffer(NU)

    function _ξ(t)
        x!(X, t)
        mul!(U, Kr(t), α(t)-_x(t))
        U .+= μ(t)
        return (X,U)
    end

    return _ξ
end


function projection2(ξ, x!, α, μ, Kr)
    for (X,U,t) in zip(ξ.x, ξ.u, ξ.t)
        x!(X, t)
        mul!(U, Kr(t), α(t)-X)
        U .+= μ(t)
    end
end
=#