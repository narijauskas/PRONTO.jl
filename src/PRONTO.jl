module PRONTO
# __precompile__(false)
using LinearAlgebra
using SciMLBase
# using DataInterpolations
using Symbolics
using Symbolics: derivative
using DifferentialEquations
# using ControlSystems # provides lqr
using MatrixEquations # provides arec

#include("trajectories.jl")
#include("model.jl")



include("timeseries.jl")
export Timeseries


include("autodiff.jl")
# export jacobian
# export hessian
#TODO: build pronto model


# t
# R,Q (for regulator)
# x0 (for projection)
# x_eq (for search direction)

# f,fx,fu
# fxx,fxu,fuu

# l,lx,lu
# ...

# p, px, pxx

# solver kw

include("regulator.jl")
include("projection.jl")
include("cost.jl")
include("search_direction.jl")
include("armijo.jl")


# update_regulator!(Kr, ξ, model)
# update_projection!(φ, ξ, Kr, model)
# Dh = update_search_direction!(ζ, φ, Kr, model)

function pronto(ξ, model)
    
    # ξ is guess
    # (X,U) = ξ
    for i in 1:model.maxiters
        @info "iteration: $i"
        # ξ -> Kr # regulator
        @info "building regulator"
        Kr = regulator(ξ..., model)

        # ξ,Kr -> φ # projection
        φ = projection(ξ..., Kr, model)

        @info "finding search direction"
        # φ,Kr -> ζ,Dh # search direction
        ζ,Dh = search_direction(φ..., Kr, model)

        # check Dh criteria -> return ξ,Kr
        @info "Dh is $Dh"
        Dh > 0 && (@warn "increased cost - quitting"; return ξ)
        -Dh < model.tol && (@info "PRONTO converged"; return ξ)
        
        @info "calculating new trajectory:"
        # φ,ζ,Kr -> γ -> ξ # armijo
        ξ = armijo_backstep(φ..., Kr, ζ..., Dh, model)

        @info "resampling solution"
        ξ = resample(ξ...,model)
    end
    # ξ is optimal (or last iteration)

    @warn "maxiters"
    return ξ
end

export pronto

end # module
