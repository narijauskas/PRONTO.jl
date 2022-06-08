module PRONTO
# __precompile__(false)
using LinearAlgebra
using SciMLBase
using SciMLBase: @def
using Symbolics
using Symbolics: derivative
using DifferentialEquations
# using ControlSystems # provides lqr
using MatrixEquations # provides arec



include("mstruct.jl")

include("interpolants.jl")
export Interpolant

include("autodiff.jl")
export autodiff
# export jacobian
# export hessian
#TODO: build pronto model
# model = autodiff(f,l,p;NX,NU)
# autodiff!(model, f,l,p;NX,NU)



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



#=
include("regulator.jl")
include("projection.jl")
include("cost.jl")
include("search_direction.jl")
include("armijo.jl")


function pronto(ξ, model)

    # declare interpolant objects
    # ξ,φ
    # declare integrators to solve them
    # declare functions (capturing above)


    # pronto loop - update integrators

end





@inline A(model,ξ,t) = model.fx(ξ.x(t),ξ.u(t))::model.fxT
@inline B(model,ξ,t) = model.fu(ξ.x(t),ξ.u(t))::model.fuT
# A = model.fx(ξ.x(t),ξ.u(t))
# B = model.fu(ξ.x(t),ξ.u(t))


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
=#
end # module
