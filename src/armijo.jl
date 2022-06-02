

# armijo_backstep:
function armijo_backstep(x,u,Kr,z,v,Dh,model)
    γ = 1
    T = last(model.t)
    
    # compute cost
    J = cost(x,u,model)
    h = J(T)[1] + model.p(x(T))
    ξ = 0

    while γ > model.β^12
        # generate estimate
        α = Timeseries(t->(x(t) + γ*z(t)))
        μ = Timeseries(t->(u(t) + γ*v(t)))
        ξ = projection(α, μ, Kr, model)

        J = cost(ξ..., model)
        g = J(T)[1] + model.p(ξ[1](T))

        # check armijo rule
        @info "armijo update: γ = $γ"
        h-g >= -model.α*γ*Dh ? (return resample(ξ...,model)) : (γ *= model.β)
        # println("γ=$γ, h-g=$(h-g)")
    end

    @warn "maxiters"
    return resample(ξ...,model)
end


function resample(x,u,model)
    x = Timeseries(t->x(t), model.t)
    u = Timeseries(t->u(t), model.t)
    return (x,u)
end