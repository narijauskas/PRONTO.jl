

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
        α = Timeseries(t->(x(t) + γ*z(t)), model.t)
        μ = Timeseries(t->(u(t) + γ*v(t)), model.t)
        ξ = projection(α, μ, Kr, model)

        J = cost(ξ..., model)
        g = J(T)[1] + model.p(ξ[1](T))

        # check armijo rule
        @info "armijo update: γ = $γ"
        h-g >= -model.α*γ*Dh ? (return ξ) : (γ *= model.β)
        # println("γ=$γ, h-g=$(h-g)")
    end

    return ξ
end
