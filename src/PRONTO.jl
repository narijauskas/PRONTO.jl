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
include("itp.jl")
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
# --------------------------------- projection --------------------------------- #



# for projection, provided Kr(t)
function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
    # FUTURE: in-place f!(dx,x,u) 
end

# η,Kr -> ξ # projection to generate stabilized trajectory
function projection(α,μ,Kr,model)
    @unpack model
    T = last(ts)

    x = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))

    #TEST: performance against just returning x_ode as x
    # x = Functor(NX) do buf,t
    #     copy!(buf, x_ode(t))
    # end

    u = Functor(NU) do buf,t
        buf .= μ(t) - Kr(t)*(x(t)-α(t))
    end

    return (x,u)
end



# --------------------------------- search direction --------------------------------- #



#FUTURE: break apart to separate functions

function search_direction(x, u, α, model, i)
    @unpack model
    T = last(ts)

    A = Functor(NX,NX) do buf,t
        fx!(buf, x(t), u(t))
    end

    # B = Functor((buf,t)->fu!(buf, x(t), u(t)), NX, NU)
    B = Functor(NX,NU) do buf,t
        fu!(buf, x(t), u(t))
    end

    a = Functor(NX) do buf,t
        lx!(buf, x(t), u(t))
    end

    b = Functor(NU) do buf,t
        lu!(buf, x(t), u(t))
    end

    Q = Functor(NX,NX) do buf,t
        lxx!(buf, x(t), u(t))
    end
   
    R = Functor(NU,NU) do buf,t
        luu!(buf, x(t), u(t))
    end

    S = Functor(NX,NU) do buf,t
        lxu!(buf, x(t), u(t))
    end

    Ko = Functor(NU,NX) do buf,P,t
        # copy!(R(t), buf)
        # inv!()
        copy!(buf, R(t)\(S(t)'+B(t)'*P))
    end

    # --------------- solve optimizer Ko --------------- #
    tx = @elapsed begin
        PT = MArray{Tuple{NX,NX},Float64}(undef) # pxx!
        pxx!(PT, α(T)) # around unregulated trajectory

        P = solve(ODEProblem(optimizer!, PT, (T,0.0), (Ko,R,Q,A)))
        
        # Ko = inv(R)\(S'+B'*P)
        Ko = Functor(NU,NX) do buf,t
            copy!(buf, R(t)\(S(t)'+B(t)'*P(t)))
        end
    end
    tinfo(i, "optimizer solved", tx)


    # --------------- solve costate dynamics vo --------------- #
    tx = @elapsed begin
        # solve costate dynamics vo
        rT = MArray{Tuple{NX},Float64}(undef)
        px!(rT, α(T)) # around unregulated trajectory
        r = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)))

        vo = Functor(NU) do buf,t
            copy!(buf, -R(t)\(B(t)'*r(t)+b(t)))
        end
    end
    tinfo(i, "costate dynamics solved", tx)


    # --------------- forward integration for search direction --------------- #
    tx = @elapsed begin
        v = Functor(NU) do buf,z,t
            mul!(buf, Ko(t), z)
            buf .*= -1
            buf .+= vo(t)
        end

        z0 = 0 .* model.x_eq
        z = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (A,B,v)))
        
        # v = -Ko(t)*z+vo(t)
        v = Functor(NU) do buf,t
            mul!(buf, Ko(t), z(t))
            buf .*= -1
            buf .+= vo(t)
        end

        ζ = (z,v)
    end
    tinfo(i, "search direction found", tx)

    # --------------- cost derivatives --------------- #
    tx = @elapsed begin
        y0 = [0;0]
        y = solve(ODEProblem(cost_derivatives!, y0, (0.0,T), (z,v,a,b,Q,S,R)))
        Dh = y(T)[1] + rT'*z(T)
        D2g = y(T)[2] + z(T)'*PT*z(T)
    end
    tinfo(i, "cost derivatives calculated", tx)
    return ζ,Dh
end


function optimizer!(dP, P, (Ko,R,Q,A), t)
    dP .= -A(t)'*P - P*A(t) + Ko(P,t)'*R(t)*Ko(P,t) - Q(t)
end

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + K(t)'*b(t)
end

function update_dynamics!(dz, z, (A,B,v), t)
    dz .= A(t)*z + B(t)*v(z,t)
end


function cost_derivatives!(dy, y, (z,v,a,b,Qo,So,Ro), t)
    dy[1] = a(t)'*z(t) + b(t)'*v(t)
    dy[2] = z(t)'*Qo(t)*z(t) + 2*z(t)'*So(t)*v(t) + v(t)'*Ro(t)*v(t)
end


# ----------------------------------- armijo ----------------------------------- #

# armijo_backstep:
function armijo_backstep(x,u,Kr,z,v,Dh,model,i)
    @unpack model
    γ = 1
    T = last(model.ts)
    
    # compute cost
    J = cost(x,u,model)
    h = J(T)[1] + model.p(x(T)) # around regulated trajectory

    while γ > model.β^12
        info(i, "armijo: γ = $γ")
        
        # generate estimate
        # MAYBE: α̂(γ,t) & move up a level

        # α̂ = x + γz
        α̂ = Functor(NX) do buf,t
            mul!(buf, γ, z(t))
            buf .+= x(t)
        end

        # μ̂ = u + γv
        μ̂ = Functor(NU) do buf,t
            mul!(buf, γ, v(t))
            buf .+= u(t)
        end

        ξ̂ = (x̂,û) = projection(α̂, μ̂, Kr, model)

        J = cost(ξ̂..., model)
        g = J(T)[1] + model.p(x̂(T))

        # check armijo rule
        h-g >= -model.α*γ*Dh ? (return ξ̂) : (γ *= model.β)
        # println("γ=$γ, h-g=$(h-g)")
    end
    @warn "armijo maxiters"
    return (x,u)
end


function stage_cost!(dh, h, (l,x,u), t)
    dh .= l(x(t), u(t))
end

function cost(x,u,model)
    T = last(model.ts)
    h = solve(ODEProblem(stage_cost!, [0], (0.0,T), (model.l,x,u)))
    return h
end
 


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


function pronto(α,μ,model)
    info("initializing")
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
    η = (α,μ)

    x = Interpolant(t->zeros(NX), ts)
    u = Interpolant(t->zeros(NU), ts)

    z = Interpolant(t->zeros(NX), ts)
    v = Interpolant(t->zeros(NU), ts)

    for i in 1:model.maxiters
        # η -> Kr # regulator
        tx = @elapsed begin
            Kr = regulator(α, μ, model)
        end
        tinfo(i, "regulator solved", tx)
        # info("allocated: $al")
        
        # η,Kr -> ξ # projection
        tx = @elapsed begin
            ξ = projection(α, μ, Kr, model)
            update!(x, ξ[1])
            update!(u, ξ[2])
            ξ = (x,u)
        end
        tinfo(i, "projection solved", tx)
        
        # ξ,Kr -> ζ # search direction
        tx = @elapsed begin
            ζ,Dh = search_direction(ξ..., α, model, i)
            update!(z, ζ[1])
            update!(v, ζ[2])
            # update!(t->ζ[1](t), z)
            # update!(t->ζ[2](t), v) #TODO: optimize this
            ζ = (z,v)
        end
        tinfo(i, "search direction found", tx)
        
        # check Dh criteria -> return η
        info(i, "Dh is $Dh")
        Dh > 0 && (@warn "increased cost - quitting"; return η)
        -Dh < model.tol && (info(as_bold("PRONTO converged")); return η)
        
        # ξ,ζ,Kr -> γ -> ξ̂ # armijo
        tx = @elapsed begin
            ξ̂ = armijo_backstep(ξ...,Kr,ζ...,Dh,model,i)
            (x̂,û) = ξ̂
            update!(α, x̂)
            update!(μ, û)
            η = (α,μ)
        end
        tinfo(i, "trajectory update found", tx)
        
    end
    @warn "maxiters"
    return nothing
end



end #module