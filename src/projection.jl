# --------------------------------- projection  η,Kr -> ξ --------------------------------- #

function projection_x(x0,α,μ,Kr,model)
    NX = model.NX; NU = model.NU; T = model.T;
    f = model.f
    x! = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))
    _x = buffer(NX)
    X = @closure (t)->(x!(_x,t); return _x)
    # X = functor((X,t)->x!(X,t), buffer(NX))
    return X
end


function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end


function projection_u(x,α,μ,Kr,model)
    NX = model.NX; NU = model.NU
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

