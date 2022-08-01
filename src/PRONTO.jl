module PRONTO
# __precompile__(false)
using LinearAlgebra
using SciMLBase
import SciMLBase: @def
using Symbolics
using Symbolics: derivative
using DifferentialEquations
using DifferentialEquations: init # silences linter
# using ControlSystems # provides lqr
using MatrixEquations # provides arec
using FastClosures
using Statistics: median

# ---------------------------- for runtime feedback ---------------------------- #
using Crayons
as_tag(str) = as_tag(crayon"default", str)
as_tag(c::Crayon, str) = as_color(c, as_bold("[$str: "))
as_color(c::Crayon, str) = "$c" * str * "$(crayon"default")"
as_bold(str) = "$(crayon"bold")" * str * "$(crayon"!bold")"
clearln() = print("\e[2K","\e[1G")

info(str) = println(as_tag(crayon"magenta","PRONTO"), str)
info(i, str) = println(as_tag(crayon"magenta","PRONTO[$i]"), str)
tinfo(i, str, tx) = println(as_tag(crayon"magenta","PRONTO[$i]"), str, " in $(round(tx*1000; digits=2)) ms")




# ---------------------------- functional components ---------------------------- #

include("mstruct.jl")
export MStruct

include("functors.jl")
export Functor

# include("interpolants.jl")
include("interpolants.jl")
export Interpolant

include("autodiff.jl")
export jacobian, hessian
export autodiff, @unpack


export guess, pronto


#MAYBE:model is a global in the module
# type ProntoModel{NX,NU}, pass buffer dimensions parametrically?
# otherwise, model holds functions and parameters


# --------------------------------- helper functions --------------------------------- #
mapid!(dest, src) = map!(identity, dest, src)
# same as: mapid!(dest, src) = map!(x->x, dest, src)

inv!(A) = LinearAlgebra.inv!(lu!(A)) # general

# LinearAlgebra.inv!(choelsky!(A)) # if SPD

include("regulator.jl")
include("projection.jl")
include("optimizer.jl")
include("costate.jl")
include("search_direction.jl")
include("cost_derivatives.jl")
include("armijo.jl")



# ----------------------------------- main loop ----------------------------------- #

# generate a guess curve between the initial state and equilibrium
guess(t, x0, xf, T) = @. (xf - x0)*(tanh((2π/T)*t - π) + 1)/2 + x0

function pronto(model)
    NX = model.NX; NU = model.NU; T = model.T; ts = model.ts;
    α = Interpolant(t->guess(t, model.x0, model.xf, T), ts)
    μ = Interpolant(t->zeros(NU), ts)
    pronto(α,μ,model)
end

function ol_dynamics!(dx, x, (f,u), t)
    dx .= f(x,u(t))
end

function pronto(μ, model)
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
    α_ode = solve(ODEProblem(ol_dynamics!, model.x0, (0,T), (model.f, μ)))
    α = Interpolant((t->α_ode(t)), ts)
    pronto(α,μ,model)
end


#TODO: split model into model/params/t
# pronto(model,t,α,μ; params)
# pronto(t,α,μ,model; params)
#params:
    # tol
    # maxiters


function pronto(α,μ,model)

    info("initializing")
    @unpack model
    # T = model.T
    
    # memory boffers
    x = Interpolant(t->zeros(NX), ts)
    u = Interpolant(t->zeros(NU), ts)

    z = Interpolant(t->zeros(NX), ts)
    v = Interpolant(t->zeros(NU), ts)

    # to track runtimes
    stats = Dict( (s=>Float64[] for s in _subroutines())...)

    # PT = buffer(NX,NX); pxx!(PT, α(T)) # P(T) around unregulated trajectory
    pxx! = model.pxx!; _PT = buffer(NX,NX)
    PT = @closure (α)->(pxx!(_PT, α(T)); return _PT)
    # PT = functor(@closure((PT,α) -> pxx!(PT, α(T))), buffer(NX,NX))

    # rT = buffer(NX); px!(rT, α(T)) # around unregulated trajectory
    px! = model.px!; _rT = buffer(NX)
    rT = @closure (α)->(px!(_rT, α(T)); return _rT)
    # rT = functor(@closure((rT,α) -> px!(rT, α(T))), buffer(NX))

    for i in 1:model.maxiters
        
        # η -> Kr # regulator
        tx = @elapsed begin
            Kr = regulator(α,μ,model)
        end
        push!(stats[:regulator], tx)
        tinfo(i, "regulator solved", tx)


        # # η,Kr -> ξ # projection
        tx = @elapsed begin
            _x = projection_x(x0,α,μ,Kr,model)
            update!(x, _x)
            _u = projection_u(x,α,μ,Kr,model)
            update!(u, _u)
        end
        push!(stats[:projection], tx)
        tinfo(i, "projection solved", tx)
        
        tx = @elapsed begin
            Ko = optimizer(x,u,PT(α),model)
        end
        push!(stats[:optimizer], tx)
        tinfo(i, "optimizer found", tx)
        
        tx = @elapsed begin
            vo = costate_dynamics(x,u,Ko,rT(α),model)
        end
        push!(stats[:costate], tx)
        tinfo(i, "costate dynamics solved", tx)
        
        tx = @elapsed begin
            _z = search_z(x,u,Ko,vo,model)
            update!(z, _z)
            _v = search_v(z,Ko,vo,model)
            update!(v, _v)
        end
        push!(stats[:search_dir], tx)
        tinfo(i, "search direction found", tx)

        # check Dh criteria -> return η
        tx = @elapsed begin
            (Dh,D2g) = cost_derivatives(x,u,z,v,rT(α),PT(α),model)
        end
        push!(stats[:cost_derivs], tx)
        tinfo(i, "cost derivatives solved", tx)
        info(i, "Dh is $Dh")
        Dh > 0 && (@warn "increased cost - quitting"; return ((α,μ),stats))
        -Dh < model.tol && (info(as_bold("PRONTO converged")); return ((α,μ),stats))
        
        
        # ξ,ζ,Kr -> γ -> ξ̂ # armijo
        tx = @elapsed begin
            (x̂,û) = armijo_backstep(x,u,z,v,Kr,Dh,i,model)
            update!(α, x̂)
            update!(μ, û)
            # η = (α,μ)
        end
        push!(stats[:trajectory], tx)
        tinfo(i, "trajectory update found", tx)
        
    end
    # @warn "maxiters"
    return ((α,μ),stats)
end

#TODO: PRONTO stats function
# print max/median/min of each section
# push!(stats[trajectory], tx)

function _subroutines()
    return [
        :regulator,
        :projection,
        :optimizer,
        :costate,
        :search_dir,
        :cost_derivs,
        :trajectory,
    ]
end

function overview(stats)
    println("\t\tminimum  ...  median  ...  maximum  ...  compile")
    for name in _subroutines()
        print("$name:\t")
        tx = stats[name]
        print(_ms(minimum(tx)), "  ...  ")
        print(_ms(median(tx)), "  ...  ")
        print(_ms(maximum(tx)), "  ...  ")
        print(_ms(tx[1]))
        println()
    end
end

_ms(tx) = "$(round(tx*1000; digits=2)) ms"


end #module