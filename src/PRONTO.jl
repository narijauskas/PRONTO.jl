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




# ----------------------------------- main loop ----------------------------------- #

# generate a guess curve between the initial state and equilibrium
guess(t, x0, x_eq, T) = @. (x_eq - x0)*(tanh((2π/T)*t - π) + 1)/2 + x0

function pronto(model)
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
    α = Interpolant(t->guess(t, model.x0, model.x_eq, T), ts)
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

#params:
    # tol
    # maxiters


function pronto(α,μ,model)

    info("initializing")
    @unpack model
    T = last(ts)
    
    # memory boffers
    x = Interpolant(t->zeros(NX), ts)
    u = Interpolant(t->zeros(NU), ts)

    z = Interpolant(t->zeros(NX), ts)
    v = Interpolant(t->zeros(NU), ts)

    # to track runtimes
    stats = Dict( (s=>Float64[] for s in _subroutines())...)

    
    A = functor(@closure((A,t) -> fx!(A,x(t),u(t))), buffer(NX,NX))
    B = functor(@closure((B,t) -> fu!(B,x(t),u(t))), buffer(NX,NU))
    a = functor(@closure((a,t) -> lx!(a,x(t),u(t))), buffer(NX))
    b = functor(@closure((b,t) -> lu!(b,x(t),u(t))), buffer(NU))
    Q = functor(@closure((Q,t) -> lxx!(Q,x(t),u(t))), buffer(NX,NX))
    R = functor(@closure((R,t) -> luu!(R,x(t),u(t))), buffer(NU,NU))
    S = functor(@closure((S,t) -> lxu!(S,x(t),u(t))), buffer(NX,NU))

    # PT = buffer(NX,NX); pxx!(PT, α(T)) # P(T) around unregulated trajectory
    PT = functor(@closure((PT) -> pxx!(PT, α(T))), buffer(NX,NX))

    # rT = buffer(NX); px!(rT, α(T)) # around unregulated trajectory
    rT = functor(@closure((rT) -> px!(rT, α(T))), buffer(NX))


    for i in 1:model.maxiters
        
        # η -> Kr # regulator
        tx = @elapsed begin
            Kr = regulator(α,μ,model)
        end
        push!(stats[:regulator], tx)
        tinfo(i, "regulator solved", tx)


        # # η,Kr -> ξ # projection
        tx = @elapsed begin
            _x = projection_x(NX,T,α,μ,Kr,f,x0)
            update!(x, _x)
            _u = projection_u(NX,NU,α,μ,Kr,x)
            update!(u, _u)
        end
        push!(stats[:projection], tx)
        tinfo(i, "projection solved", tx)
        
        tx = @elapsed begin
            Ko = optimizer(A,B,Q,R,S,PT(),NX,NU,T)
        end
        push!(stats[:optimizer], tx)
        tinfo(i, "optimizer found", tx)
        
        tx = @elapsed begin
            vo = costate_dynamics(Ko,A,B,a,b,R,rT(),NX,NU,T)
        end
        push!(stats[:costate], tx)
        tinfo(i, "costate dynamics solved", tx)
        
        tx = @elapsed begin
            _z = search_z(NX,T,Ko,vo,A,B)
            update!(z, _z)
            _v = search_v(NU,z,Ko,vo)
            update!(v, _v)
        end
        push!(stats[:search_dir], tx)
        tinfo(i, "search direction found", tx)

        # check Dh criteria -> return η
        tx = @elapsed begin
            (Dh,D2g) = cost_derivatives(z,v,a,b,Q,S,R,rT(),PT(),T)
        end
        push!(stats[:cost_derivs], tx)
        tinfo(i, "cost derivatives solved", tx)
        info(i, "Dh is $Dh")
        Dh > 0 && (@warn "increased cost - quitting"; return ((α,μ),stats))
        -Dh < model.tol && (info(as_bold("PRONTO converged")); return ((α,μ),stats))
        
        
        # ξ,ζ,Kr -> γ -> ξ̂ # armijo
        tx = @elapsed begin
            (x̂,û) = armijo_backstep(x,u,Kr,z,v,Dh,i,f,l,p,x0,NX,NU,T)
            update!(α, x̂)
            update!(μ, û)
            # η = (α,μ)
        end
        push!(stats[:trajectory], tx)
        tinfo(i, "trajectory update found", tx)
        
    end
    # @warn "maxiters"
    return nothing
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

# ----------------------------------- armijo ----------------------------------- #

# armijo_backstep:
function armijo_backstep(x,u,Kr,z,v,Dh,i,f,l,p,x0,NX,NU,T; aα=0.4, aβ=0.7)
    γ = 1
    
    # compute cost
    J = cost(x,u,l,T)
    h = J(T)[1] + p(x(T)) # around regulated trajectory

    while γ > aβ^12
        info(i, "armijo: γ = $γ")
        
        # generate estimate
        # MAYBE: α̂(γ,t) & move up a level

        # α̂ = x + γz
        α̂ = functor(buffer(NX)) do X,t
            mul!(X, γ, z(t))
            X .+= x(t)
        end

        # μ̂ = u + γv
        μ̂ = functor(buffer(NU)) do U,t
            mul!(U, γ, v(t))
            U .+= u(t)
        end
        
        x̂ = projection_x(NX,T,α̂,μ̂,Kr,f,x0)
        û = projection_u(NX,NU,α̂,μ̂,Kr,x̂)

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
 


end #module