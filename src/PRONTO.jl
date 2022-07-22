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



# ---------------------------- functional components ---------------------------- #


include("mstruct.jl")
export MStruct

include("functors.jl")
export Functor

#MAYBE: interpolant knows element size/type
#MAYBE: just write simple custom interpolant?
include("interpolants.jl")
export Interpolant

include("autodiff.jl")
export autodiff, unpack
# export jacobian
# export hessian
# model = autodiff(f,l,p;NX,NU)
# autodiff!(model, f,l,p;NX,NU)



#YO: #TODO: model is a global in the module
# type ProntoModel{NX,NU}, pass buffer dimensions parametrically?
# otherwise, model holds functions and parameters


export guess, pronto






# --------------------------------- ode functions --------------------------------- #
#FUTURE: @unpack model

# for regulator
function riccati!(dP, P, (fx!,fu!,Qr,Rr,iRr,X_α,U_μ,NX,NU), t)
    Ar = MArray{Tuple{NX,NX},Float64}(undef)
    Br = MArray{Tuple{NX,NU},Float64}(undef)
    iRrBr = MArray{Tuple{NU,NX},Float64}(undef)
    Kr = MArray{Tuple{NU,NX},Float64}(undef)

    # in-place update of buffers (Ar, Br, Kr) for time t
    fx!(Ar, X_α(t), U_μ(t)) # Ar = fx(α(t), μ(t))
    fu!(Br, X_α(t), U_μ(t)) # Br = fu(α(t), μ(t))
    mul!(iRrBr, iRr(t), Br') # Kr = inv(Rr(t))*Br'*P
    mul!(Kr, iRrBr, P)

    #TEST: hopefully optimized by compiler?
    # if not, do each step inplace to local buffers or SVectors
    dP .= -Ar'P - P*Ar + Kr'*Rr(t)*Kr - Qr(t)
end




# for projection, provided Kr(t)
function stabilized_dynamics!(dx, x, (f,fu!,Kr,α,μ), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
    # FUTURE: in-place f!(dx,x,u) 
end




#TODO:
function optimizer!(dP, P, (A,B,Q,R,S), t)
    #TODO: buffers Ko, R, S, B, Q, A
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end




# ------------------------------ helper functions ------------------------------ #

mapid!(dest, src) = map!(identity, dest, src)
# same as:
# mapid!(dest, src) = map!(x->x, dest, src)


# update each X(t) by re-solving the ode from x0
function update!(X::Interpolant, ode, x0)
    reinit!(ode,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(ode, X.itp.t))
        X[i] .= x
        # map!(v->v, X[i], x)
    end

    #TEST:
    # for (Xi, (x,t)) in zip(X,TimeChoiceIterator(ode, X.itp.t))
    #     map!(v->v, Xi, x)
    # end
    return nothing
end



# update_Kr!(Kr, X_α, U_μ, model) = update_Kr!(Kr, model.fx!, model.fu!, model.Qr, model.Rr, model.iRr, X_α, U_μ, model)

# solve for regulator
# ξ or φ -> Kr
# update the values stored in Kr
#TODO: how this works
# solve algebraic riccati for P(T)
# 
# function update_Kr!(Kr,fx!,fu!,Qr,Rr,iRr,X_α,U_μ,model)
function update_Kr!(Kr, X_α, U_μ, model)
    @unpack model
    T = last(ts)
    # ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU

    Ar = MArray{Tuple{NX,NX},Float64}(undef)
    Br = MArray{Tuple{NX,NU},Float64}(undef)
    iRrBr = MArray{Tuple{NU,NX},Float64}(undef)

    fx!(Ar, X_α(T), U_μ(T))
    fu!(Br, X_α(T), U_μ(T))

    PT,_ = arec(Ar, Br*iRr(T)*Br', Qr(T))
    Pr = solve(ODEProblem(riccati!, PT, (T,0.0), (fx!,fu!,Qr,Rr,iRr,X_α,U_μ,NX,NU)))
    
    # Kr(t) = inv(Rr(t))*Br(t)'*Pr(t)
    for (Kr_t, t) in zip(Kr, times(Kr))
        fu!(Br, X_α(t), U_μ(t))
        mul!(iRrBr, iRr(t), Br') # {NU,NU}*{NU,NX}->{NU,NX}
        mul!(Kr_t, iRrBr, Pr(t))
    end
    # return Kr, Pr
end


# φ,Kr -> ξ # projection
function update_ξ!(X_x,U_u,Kr,X_α,U_μ,model)
    @unpack model
    T = last(ts)
    X_ode = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (f,fu!,Kr,X_α,U_μ)))
    for (X, U, t) in zip(X_x, U_u, times(X_x))
        X .= X_ode(t)
        U .= U_μ(t) - Kr(t)*(X-X_α(t))
    end
end




guess(t, x0, x_eq, T) = @. (x_eq - x0)*(tanh((2π/T)*t - π) + 1)/2 + x0






function search_direction(X_z, U_v, X_x, U_u, model)
    @unpack model
    T = last(ts)
    # ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
    # fx! = model.fx!; fu! = model.fu!
    # lxx! = model.lxx!; luu! = model.luu!; lxu! = model.lxu!;
    # px! = model.px!; pxx! = model.pxx!
    
    # buffers
    Ko = MArray{Tuple{NU,NX},Float64}(undef)
    vo = MArray{Tuple{NU},Float64}(undef)

    A = MArray{Tuple{NX,NU},Float64}(undef) # fx!
    B = MArray{Tuple{NX,NU},Float64}(undef) # fu!
    Q = MArray{Tuple{NX,NX},Float64}(undef) # lxx!
    R = MArray{Tuple{NU,NU},Float64}(undef) # luu!
    S = MArray{Tuple{NX,NU},Float64}(undef) # lxu!
    
    #TODO: A,B,Q,R,S or pass equivalent functions
    fx!(A, X_x(t), U_u(t))  # NX,NU
    fu!(B, X_x(t), U_u(t))  # NX,NU
    lxx!(Q, X_x(t), U_u(t)) # NX,NX
    luu!(R, X_x(t), U_u(t)) # NU,NU
    lxu!(S, X_x(t), U_u(t)) # NX,NU

    PT = MArray{Tuple{NX,NX},Float64}(undef)
    pxx!(PT, model.x_eq)
    # solve optimizer Ko
    P = Timeseries(solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S))))
    for (Ko_t, t) in zip(Ko, times(X_x))
        # update R,S,B
        fu!(B, X_x(t), U_u(t))
        luu!(R, X_x(t), U_u(t))
        lxu!(S, X_x(t), U_u(t))
        
        #NU,NU*NU,NX
        #TODO: inplace operators?
        Ko_t .= R\(S'+B'*P(t))
    end

    # NU,NU * NU,1
    #

    # solve costate dynamics vo


    # forward integration, solve z
    # update X_z
    # update U_v

    # ζ is updated
end














# ----------------------------------- main loop ----------------------------------- #

function pronto(model)
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
    X_x = Interpolant(t->guess(t, model.x0, model.x_eq, T), ts, NX)
    U_u = Interpolant(ts, NU)
    pronto(X_x,U_u,model)
end

function pronto(X_x,U_u,model)
    @info "initializing"
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU

    # X_x = Interpolant(ts, NX)
    X_α = Interpolant(t->X_x(t), ts, NX)
    X_z = Interpolant(ts, NX)
    # U_u = Interpolant(ts, NU)
    U_μ = Interpolant(t->U_u(t), ts, NU)
    U_v = Interpolant(ts, NU)

    # Pr = Interpolant(ts, NX, NX)
    Kr = Interpolant(ts, NU, NX)

    # buffers
    #MAYBE: generalize T beyond F64?
    Ar = MArray{Tuple{NX,NX},Float64}(undef)
    # model.fx!(Ar, X_α(t), U_μ(t))
    Br = MArray{Tuple{NX,NU},Float64}(undef)
    # model.fu!(Br, X_α(t), U_μ(t))
    # Kr = MArray{Tuple{NU,NX},Float64}(undef)
    # mul!(Kr, iRr(t)*Br', Pr(t))
    PT = MArray{Tuple{NX,NX},Float64}(undef)



    # ode solver for Pr(t)
    fx! = model.fx!; fu! = model.fu!
    Qr = model.Qr; Rr = model.Rr; iRr = model.iRr;

    fx!(Ar, model.x_eq, model.u_eq)
    fu!(Br, model.x_eq, model.u_eq)
    P,_ = arec(Ar, Br*iRr(T)*Br', Qr(T)); PT .= P



    # ξ is guess
    # (X,U) = ξ
    for i in 1:model.maxiters
        @info "iteration: $i"
        # ξ or φ -> Kr # regulator
        # Kr = regulator(φ..., model)
        tx = @elapsed update_Kr!(Kr, X_α, U_μ, model)
        @info "(itr: $i) regulator solved in $tx seconds"

        # φ,Kr -> ξ # projection
        # φ = projection(ξ..., Kr, model)
        tx = @elapsed update_ξ!(X_x,U_u,Kr,X_α,U_μ,model)
        @info "(itr: $i) projection solved in $tx seconds"

        # temporary:
        update!(t->X_x(t), X_α)
        update!(t->U_u(t), U_μ)

        # @info "finding search direction"
        # # φ,Kr -> ζ,Dh # search direction
        # ζ,Dh = search_direction(φ..., Kr, model)
        # @info "(itr: $i) search direction found in $tx seconds"

        # # check Dh criteria -> return ξ,Kr
        # @info "Dh is $Dh"
        # Dh > 0 && (@warn "increased cost - quitting"; return ξ)
        # -Dh < model.tol && (@info "PRONTO converged"; return ξ)
        
        # @info "calculating new trajectory:"
        # # φ,ζ,Kr -> γ -> ξ # armijo
        # ξ = armijo_backstep(φ..., Kr, ζ..., Dh, model)

    end
    # ξ is optimal (or last iteration)

    # @warn "maxiters"
    # return ξ

end





end #module