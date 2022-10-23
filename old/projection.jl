# --------------------------------- projection  η,Kr -> ξ --------------------------------- #

# function projection!(x,u,α,μ,Kr,x0,model)
#     NX = model.NX; NU = model.NU; T = model.T;
#     f = model.f
#     x! = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))
#     _x = Buffer{Tuple{NX}}()
#     x = @closure (t)->(x!(_x,t); return _x)
# end








function projection_x(x0,α,μ,Kr,model)
    NX = model.NX; NU = model.NU; T = model.T;
    f! = model.f!
    x! = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f!)))
    _x = Buffer{Tuple{NX}}()
    # x(t) = (x!(_x,t); return _x)
    x = @closure (t)->(x!(_x,t); return _x)
    return x
end


function stabilized_dynamics!(dx, x, (α,μ,Kr,f!), t)
    u = μ(t) - Kr(t)*(x-α(t))
    f!(dx,x,u)
end


function projection_u(x,α,μ,Kr,model)
    NX = model.NX; NU = model.NU
    _u = Buffer{Tuple{NU}}()
    _x = Buffer{Tuple{NX}}()
    function u(t)
        # u = μ - Kr*(x-α)
        copy!(_x, α(t))
        _x .-= x(t)
        mul!(_u, Kr(t), _x)
        _u .+= μ(t)
        return _u
    end
    return u
end

