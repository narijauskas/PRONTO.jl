# module PRONTO
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



# ---------------------------- functional components ---------------------------- #


include("mstruct.jl")
export MStruct

include("functors.jl")
export Functor

include("interpolants.jl")
export Interpolant

#TODO: model type?
include("autodiff.jl")
export autodiff
# export jacobian
# export hessian
# model = autodiff(f,l,p;NX,NU)
# autodiff!(model, f,l,p;NX,NU)







# ------------------------------ ODE interface ------------------------------ #


#include("integration.jl")
# reinitialize integrator ig from x0, and re-solve, saving steps to interpolant X
function resolve!(ig,x0,X::Interpolant)
    reinit!(ig,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(ig, X.itp.t))
        X[i] .= x
        # map!(v->v, X[i], x)
    end
    return nothing
end

# to use:
# 1. make ODEProblem
# 2. initialize as integrator ig
# 3. make interpolant
# 4. resolve as needed, storing to interpolant


#FUTURE: for convenience,
# struct Integrator
#     X::Interpolant
#     ig # initialized ODE integrator
# end












# --------------------------------- functions --------------------------------- #


# non-capturing functions
function riccati!(dP, P, (Ar,Br,Kr,fx!,fu!,Qr,Rr,X_α,U_μ), t)
    # update buffers in-place
    fx!(Ar, X_α(t), U_μ(t))
    fu!(Br, X_α(t), U_μ(t))
    mul!(Kr, inv(Rr(t))*Br', P)
    dP .= -Ar'P - P*Ar + Kr'*Rr(t)*Kr - Qr(t)
    # K = inv(R(t))*B(t)'*P # instantenously evaluated K
    # dP .= -A(t)'P - P*A(t) + K'*R(t)*K - Q(t)
end

function stabilized_dynamics!(dx,x,(f,Kr,α,μ),t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
end


test_Ar(Ar) = Ar(1.3)



P = copy(PT)
dP = copy(P)
riccati!(dP,P,(Ar,Br,Kr,model.fx!,model.fu!,model.Qr,model.Rr,X_α,U_μ),1.3)
@code_warntype riccati!(dP,P,(Ar,Br,Kr,model.fx!,model.fu!,model.Qr,model.Rr,X_α,U_μ),1.3)

# @def Ar (t->model.fx(x(t),u(t)))
# (@Ar)(t)

# pronto!(x,u,α,μ,model)
# pronto!(x,u,model)

# ----------------------------------- main loop ----------------------------------- #
function pronto(model)
    T = last(model.ts); NX = model.NX; NU = model.NU

    # maybe? define arctan guess trajectory from x0 to xeq
    X_x = Interpolant(model.ts, model.NX)
    X_x[end] = model.x_eq
    X_α = Interpolant(model.ts, model.NX)
    X_z = Interpolant(model.ts, model.NX)
    U_u = Interpolant(model.ts, model.NU)
    U_u[end] = model.u_eq
    U_μ = Interpolant(model.ts, model.NU)
    U_v = Interpolant(model.ts, model.NU)
    Pr = Interpolant(model.ts, model.NX, model.NX)

    # buffers
    Ar = MArray{Tuple{NX,NX},Float64}(undef) #FIX: generalize T beyond F64?
    # model.fx!(Ar, X_α(t), U_μ(t))
    Br = MArray{Tuple{NX,NU},Float64}(undef)
    Kr = MArray{Tuple{NU,NX},Float64}(undef)
    # mul!(buf, inv(Rr(t))*Br(t)', Pr(t))
    PT = MArray{Tuple{NX,NX},Float64}(undef)

    # update buffers in-place (use x_eq?)
    model.fx!(Ar, X_α(T), U_μ(T))
    model.fu!(Br, X_α(T), U_μ(T))
    P,_ = arec(Ar, Br*inv(model.Rr(T))*Br', model.Qr(T))
    PT .= P
    # end

    Pr_ode = init(
        ODEProblem(riccati!, PT, (T,0.0),(Ar,Br,Kr,model.fx!,model.fu!,model.Qr,model.Rr,X_α,U_μ)),
        Tsit5()
    )

    #     # φ->Kr
    #     # update_Kr!()
    @info "solving regulator"
    # update terminal cost

    model.fx!(Ar, X_α(T), U_μ(T))
    model.fu!(Br, X_α(T), U_μ(T))
    P,_ = arec(Ar, Br*inv(model.Rr(T))*Br', model.Qr(T))
    PT .= P

    resolve!(Pr_ode, PT, Pr)
    # re-solve riccati
    # reinit!(Pr_ode, PT)
    # for (i,(x,t)) in enumerate(TimeChoiceIterator(Pr_ode, Pr.itp.t))
    #     Pr[i] .= x
    #     # map!(v->v, X[i], x)
    # end
    
    @info "regulator solved"
    # update Kr via:
    # mul!(Kr, inv(Rr(t))*Br', P)

    
    #     # φ,Kr->ξ
    #     # @info "projection"
    #     # update_ξ!()
    
    #     # ξ,Kr->ζ # search direction
    
    #     # ζ->Dh # cost derivatives
    #     # exit criteria -> return ξ
    
    #     # γ # armijo sub-loop
    #     # ξ,ζ,γ->φ # new estimate
    #     # φ,Kr->ξ # projection
    # end
end

pronto(model)
@time pronto(model)

# const Pr = Interpolant((t)->PT(T), model.ts, model.NX, model.NX)
test_resolve!(Pr_ode, PT, Pr) = resolve!(Pr_ode, PT, Pr)
# @def Ar model.fx(X_α(t),U_μ(t))
# foo = t->model.fx(X_α(t),U_μ(t))
# function pronto(model, α0, μ0)
    # φ/ξ0/ξ is initial guess
    # φ->Kr # regulator

    # @pronto_setup

    # X_α always starts at x0 and ends at x_eq
    # α0 should be arctan guess

# core data storage
# const X_x = Interpolant(t->α0(t), model.ts, model.NX)
# const X_α = Interpolant(t->α0(t), model.ts, model.NX)


# A!(buf,model,x,u,t) = model.fx!(buf,x(t),u(t))


# A = Functor(NX,NX) do 
#     (buf,t)->model.fx!(buf, X_x(t), U_u(t))
# end

# B = Functor(NU,NU) do 
#     (buf,t)->model.fu!(buf, X_x(t), U_u(t))
# end

    # # core functions
    # A(x,u,t) = model.fx(x(t),u(t))
    # B(x,u,t) = model.fu(x(t),u(t))

    # Ar = (t)->A(X_α,U_μ,t) # captures (X_α) and (U_μ)
    # Br = (t)->B(X_α,U_μ,t) # captures (X_α) and (U_μ)



# invRr = model.invRr

# PT,_ = arec(Ar(T), Br(T), Rr(T), Qr(T))

# PT will always be around x_eq/u_eq



# # beautiful:
# @code_warntype model.fx!(buf, X_α(t), U_μ(t))
# @code_warntype model.fu!(buf, X_α(t), U_μ(t))
# @code_warntype Br(t)
# @benchmark Br(t)
# @benchmark model.fu!(buf, X_α(t), U_μ(t))


@code_native model.fx!(buf, X_α(t), U_μ(t))
@code_native model.fu!(buf, X_α(t), U_μ(t))

@profview foreach(t->Br(t), model.ts)
@report_opt Ar(t)

@report_opt resolve!(Pr_ode3, PT(T), Pr)



    # regulator
# const KrT = inv(Rr(T))*Br(T)'*PT
# const Kr = Interpolant((t)->KrT, model.ts)
    # Kr = Interpolant(t->zeros(model.NU,model.NX), model.ts)
    # Kr(t) = inv(Rr(t))*B(X_α,U_μ,t)'*Pr(t) # captures Pr, (X_α) and (U_μ)

    # update the value of Kr (by solving Pr) using X_α,U_μ
function update_Kr!()
    reinit!(Pr_ode,PT)
    for (i,(Pr,t)) in enumerate(TimeChoiceIterator(Pr_ode, Kr.t))
        Kr[i] = invRr(t)*model.fu(X_α(t),U_μ(t))'*Pr
    end
    # map!((Pr,t)->(invRr(t)*model.fu(X_α(t),U_μ(t))'*Pr),Kr.u, TimeChoiceIterator(Pr_ode, Kr.t))
    # resolve!(Pr_ode, PT, Pr)
    # update!(t->inv(Rr(t))*Br(t)'*Pr(t), Kr)
    return nothing
end 

  

const ode2 = ODEProblem(stabilized_dynamics!, model.x0, (0.0,T), (model.f,Kr,X_α,U_μ))
const X_x_ode = init(ode2, Tsit5())

# resample and save a new (X_x) and (U_u)
function update_ξ!()
    resolve!(X_x_ode, model.x0, X_x)
    update!(t->(U_μ(t) - Kr(t)*(X_x(t)-X_α(t))), U_u)
    return nothing
end



function pronto_loop()
for i in 1:model.maxiters

    # φ->Kr
    # @info "regulator update"
    update_Kr!()

    # φ,Kr->ξ
    # @info "projection"
    update_ξ!()

    # ξ,Kr->ζ # search direction

    # ζ->Dh # cost derivatives
    # exit criteria -> return ξ

    # γ # armijo sub-loop
    # ξ,ζ,γ->φ # new estimate
    # φ,Kr->ξ # projection
end
end


    # @warn "maxiters"
    # return (X_x,U_u)
# end




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
# export pronto
# end # module
