
# ξ = pronto(ξ, model...)

#= ξ is guess
 for i in 1:maxiters
    @info "iteration: $i"
    ξ -> Kr # regulator
    ξ,Kr -> φ # projection
    φ,Kr -> ζ,Dh # search direction

    # check Dh criteria -> return ξ,Kr
    
    @info "Dh is [...], doing x"

    φ,ζ,Kr -> γ -> ξ # armijo
end
 ξ is optimal (or last iteration)
=#

function pronto(ξ, model...; kw...)
        
    # ξ is guess
    for i in 1:maxiters
        @info "iteration: $i"
        #TODO:
        ξ -> Kr # regulator
        #TODO:
        ξ,Kr -> φ # projection
        #TODO:
        φ,Kr -> ζ,Dh # search direction

        # check Dh criteria -> return ξ,Kr
        @info "Dh is $Dh"
        Dh > 0 && (@warn "increased cost from update direction"; return ξ)
        -Dh < 1e-2 && (@info "PRONTO converged"; return ξ)
        
        @info "running armijo rule:"
        #TODO:
        φ,ζ,Kr -> γ -> ξ # armijo
    end
    # ξ is optimal (or last iteration)

    @warn "maxiters"
    return ξ
end





# Kr = regulator(...)
# φ = projection(...)
# ζ, Dh = search_direction(...)
# ξ = 