
# ----------------------------------- armijo  ξ,ζ,Kr -> γ -> ξ̂ ----------------------------------- #

# armijo_backstep:
function armijo_backstep(x,u,z,v,Kr,Dh,i,model; aα=0.4, aβ=0.7)
    NX = model.NX; NU = model.NU; T = model.T;
    f = model.f; l = model.l; p = model.p; x0 = model.x0;
    γ = 1
    
    # compute cost
    J = cost(x,u,l,T)
    h = J(T)[1] + p(x(T)) # around regulated trajectory

    while γ > aβ^12
        info(i, "armijo: γ = $γ")
        # generate estimate

        # α̂ = x + γz
        _α̂ = Buffer{Tuple{NX}}()
        function α̂(t)
            mul!(_α̂, γ, z(t))
            _α̂ .+= x(t)
        end

        # μ̂ = u + γv
        _μ̂ = Buffer{Tuple{NU}}()
        function μ̂(t)
            mul!(_μ̂, γ, v(t))
            _μ̂ .+= u(t)
        end

        x̂ = projection_x(x0,α̂,μ̂,Kr,model)
        û = projection_u(x̂,α̂,μ̂,Kr,model)

        J = cost(x̂,û,l,T)
        g = J(T)[1] + p(x̂(T))

        # check armijo rule
        h-g >= -aα*γ*Dh ? (return (x̂,û)) : (γ *= aβ)
        # println("γ=$γ, h-g=$(h-g)")
    end
    @warn "armijo maxiters"
    return (x,u)
end


function stage_cost!(dh, h, (l,x,u), t)
    dh .= l(x(t), u(t))
end

function cost(x,u,l,T)
    h = solve(ODEProblem(stage_cost!, [0], (0.0,T), (l,x,u)))
    return h
end
 