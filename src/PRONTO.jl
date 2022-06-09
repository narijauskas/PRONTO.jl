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


include("mstruct.jl")

include("interpolants.jl")
export Interpolant

include("autodiff.jl")
export autodiff
# export jacobian
# export hessian
# model = autodiff(f,l,p;NX,NU)
# autodiff!(model, f,l,p;NX,NU)



#include("integration.jl")
# reinitialize integrator ig from x0, and solve, saving steps to iterator X
function resolve!(ig,x0,X)
    reinit!(ig,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(ig, X.t))
        X[i] = x
    end
    return nothing
end


#FUTURE: for convenience,
# struct Integrator
#     X::Interpolant
#     ig # initialized ODE integrator
# end


function set_params!(model)
    model.maxiters = 10
    model.ts = 0:0.01:model.T
end




# non-capturing functions
function riccati!(dP, P, (A,B,Q,R), t)
    K = inv(R(t))*B(t)'*P # instantenously evaluated K
    dP .= -A(t)'P - P*A(t) + K'*R(t)*K - Q(t)
end






# --------------------------- main loop --------------------------- #


function pronto(model, α0, μ0)
    # φ/ξ0/ξ is initial guess
    # φ->Kr # regulator

    # @pronto_setup

    # core data storage
    X_x = Interpolant(t->α0(t), model.ts)
    X_α = Interpolant(t->α0(t), model.ts)
    X_z = Interpolant(t->zeros(model.NX), model.ts)

    U_u = Interpolant(t->μ0(t), model.ts)
    U_μ = Interpolant(t->μ0(t), model.ts)
    U_v = Interpolant(t->zeros(model.NU), model.ts)

    # core functions
    A(x,u,t) = model.fx(x(t),u(t))
    B(x,u,t) = model.fu(x(t),u(t))

    
    Ar(t) = A(X_α,U_μ,t) # captures (X_α) and (U_μ)
    Br(t) = B(X_α,U_μ,t) # captures (X_α) and (U_μ)
    Qr(t) = model.Qr(t)
    Rr(t) = model.Rr(t)
    Kr(t) = inv(Rr(t))*B(X_α,U_μ,t)'*Pr(t) # captures Pr, (X_α) and (U_μ)

    T = last(model.ts)
    # regulator
    function Pr_T(x,u)
        T = last(model.ts)
        PT,_ = arec(Ar(T), Br(T)*inv(Rr(T))*Br(T)', Qr(T))
        return PT
    end

    Pr = Interpolant((t)->Pr_T(X_α,U_μ), model.ts)

    ode = ODEProblem(riccati!, Pr_T(X_α,U_μ), (T,0.0), (Ar,Br,Qr,Rr))
    Pr_ode = init(ode, Tsit5())

    # update the value of Kr (by updating Pr) using X_α,U_μ
    function update_Kr!()
        resolve!(Pr_ode, Pr_T(X_α,U_μ), Pr)
        return nothing
    end

    function stabilized_dynamics!(dx,x,(Kr,α,μ),t)
        u = μ(t) - Kr(t)*(x-α(t))
        dx .= model.f(x,u)
    end

    ode = ODEProblem(stabilized_dynamics!, model.x0, (0.0,T), (Kr,X_α,U_μ))
    X_x_ode = init(ode, Tsit5())

    # resample and save a new (X_x) and (U_u)
    function update_ξ!()
        resolve!(X_x_ode, model.x0, X_x)
        update!(t->(U_μ(t) - Kr(t)*(X_x(t)-X_α(t))), U_u)
        return nothing
    end

    for i in 1:model.maxiters

        # φ->Kr
        @info "regulator update"
        update_Kr!()

        # φ,Kr->ξ
        @info "projection"
        update_ξ!()

        # ξ,Kr->ζ # search direction

        # ζ->Dh # cost derivatives
        # exit criteria -> return ξ

        # γ # armijo sub-loop
        # ξ,ζ,γ->φ # new estimate
        # φ,Kr->ξ # projection
    end
    


    @warn "maxiters"
    return (X_x,U_u)
end




# --------------------------- core data storage --------------------------- #




# --------------------------- core functions --------------------------- #

# useful functions
# @inline A(model,ξ,t) = model.fx(ξ.x(t),ξ.u(t))
# @inline B(model,ξ,t) = model.fu(ξ.x(t),ξ.u(t))

# arbitrary guess trajectory
# φ = α,μ unstabilized trajectory
# ξ = x,u stabilized trajectory (depends on ξ)

# @def pronto_setup begin ... end

# solving the regulator

# A(tj) = (t)->A(tj,t) # function which creates a function of t with captured trajectory tj
# alternatively, Ar(t) = A(φ,t), captures φ

# B(tj) = (t)->B(tj,t) # function which creates a function of t with captured trajectory tj



# --------------------------- regulator --------------------------- #


# --------------------------- projection --------------------------- #





# --------------------------- update values --------------------------- #

# map ξ0 -> Kr
# integrate Pr from PT to 0
# Kr is a function of Pr


# now Kr is up to date






#option 2: define macros, or functions at global scope






# @expand model A B
# # likely need to escape A,B
# A = model.A
# B = model.B

# @inline A(model,ξ,t) = model.fx(ξ.x(t),ξ.u(t))
# @inline B(model,ξ,t) = model.fu(ξ.x(t),ξ.u(t))

# Kr(model,t) = inv(model.Rr(t))*B(model,modelt)'*P




# --------------------------- earlier versions and comments --------------------------- #







# @def Rr model.Rr
# (@Rr)(t) instead of model.Rr(t)
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

=#
export pronto
end # module
