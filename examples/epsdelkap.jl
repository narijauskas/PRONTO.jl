# function σ_κ(s,κ)
# # Saturation (hockeystick) function used in PRONTO central path constraint implementation
# # - Used to reshape barrier function for non-convex constraints to ensure rapid falloff
# #   outside of constraint keepout region.

#     if (s > 0)
#         return tanh(s * κ)
#     else
#         return (s * κ)
#     end

# end

using IfElse

"""
Saturation (hockeystick) function used in PRONTO central path constraint implementation
- Used to reshape barrier function for non-convex constraints to ensure rapid falloff
  outside of constraint keepout region.
"""
σ_κ(s,κ) = IfElse.ifelse(s > 0, tanh(s * κ), s * κ)
# σ_κ(s,κ) = (s > 0) ? tanh(s * κ) : (s * κ)

function β_δ(s,δ)
# Finite barrier function used in PRONTO solver to allow finite cost in infeasible regions

    # Useful constants (included to allow adjustment of order)
    K   = 2
    K1	= K - 1

    # Evaluate function
    # if (s ≥ δ)
    #     return -log(s)
    # else
    #     return (K1/K)*( ((s-K*δ)/(K1*δ))^K - 1 ) - log(δ)
    # end

    return IfElse.ifelse(
        s ≥ δ,
        -log(s),
        (K1/K)*( ((s-K*δ)/(K1*δ))^K - 1 ) - log(δ))
end


function BarrierFun(s,δ,κ)
# Computes a c2 smoothed barrier function for use in the PRONTO solver. This function (β_δ in 
# the supporting literature) is a C2 smoothing of the log barrier function given by:
#  β_δ(s) = /                                           - log(s), s ≤ δ
#           \  ((k-1)/k)*( ((s - k*δ)/(δ*(k-1)))^k - 1) - log(δ), s < δ
# Unlike the classic log-barrier function, this adaptation maps infeasible (negative) inputs 
# to finite (but quite high) cost depending on the parameter δ. This barrier function allows 
# feasible trajectories to be properly handled by the projection operator and PRONTO solver.
# 
# Additionally, the argument κ controls the enabling and scaling of an input saturation
# function σ_κ() which should be included on nonconvex cost functions to ensure rapid 
# falloff outside of constraint exclusion region:
#   σ(κ,z) = /  tanh(z*κ), z ≥ 0
#            \       z*κ , z < 0
# These two functions are nested in the form β_{δ,κ}(z) = β_δ(σ_κ( -c(x,u) )) for constraints 
# of the form c(x,u) ≤ 0

    # Compute composite function
    if(κ == 0)
        # Default (unscaled) beta function
        return( β_δ(    s   , δ ) )
    else
        # Default sigma-scaled beta function
        return( β_δ(σ_κ(s,κ), δ ) )
    end

end



# ## DEBUG
# using GLMakie

# ##
# # Testing delta variation
# s = (-0.25):0.05:1.25
# x = [[BarrierFun(t,0,0) for t in s],
#      [BarrierFun(t,0.25,0) for t in s],
#      [BarrierFun(t,0.50,0) for t in s]]

# # Testing kappa variation
# s = (0.001):0.01:1.25
# x = [[BarrierFun(t,0,1) for t in s],
#      [BarrierFun(t,0,4) for t in s],
#      [BarrierFun(t,0,20) for t in s]]

# fig = Figure()
# ax = Axis(fig[1,1]; xlabel="s", ylabel="β")
# foreach(i->lines!(ax, s, x[i]), 1:3)
# display(fig)