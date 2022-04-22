
# pronto!(ξ, model...)

# ξ is guess
for i in 1:maxiters
    ξ -> Kr # regulator
    ξ,Kr -> φ # projection
    φ,Kr -> ζ,Dh # search direction

    # check Dh criteria -> return ξ,Kr

    φ,ζ,Kr -> γ -> ξ # armijo
end
# ξ is optimal (or last iteration)