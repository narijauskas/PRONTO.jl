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
export autodiff, @unpack
# export jacobian
# export hessian
# model = autodiff(f,l,p;NX,NU)
# autodiff!(model, f,l,p;NX,NU)



#YO: #TODO: model is a global in the module
# type ProntoModel{NX,NU}, pass buffer dimensions parametrically?
# otherwise, model holds functions and parameters


export guess, pronto







# --------------------------------- helper functions --------------------------------- #


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




guess(t, x0, x_eq, T) = @. (x_eq - x0)*(tanh((2π/T)*t - π) + 1)/2 + x0



# --------------------------------- regulator --------------------------------- #

# for regulator
function riccati!(dP, P, (Ar,Br,Rr,Qr,Kr), t)
    #TEST: hopefully optimized by compiler?
    # if not, do each step inplace to local buffers or SVectors
    dP .= -Ar(t)'P - P*Ar(t) + Kr(P,t)'*Rr(t)*Kr(P,t) - Qr(t)
end


# solve for regulator
# ξ or φ -> Kr
function regulator(η, model)
    @unpack model
    T = last(ts)
    # ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU

    (α,μ) = η

    Ar = Functor(NX,NX) do buf,t
        fx!(buf, α(t), μ(t))
    end

    Br = Functor(NX,NU) do buf,t
        fu!(buf, α(t), μ(t))
    end

    iRrBr = Functor(NU,NX) do buf,t
        mul!(buf, iRr(t), Br(t)')
    end

    # Kr = inv(Rr)*Br'*P
    Kr = Functor(NU,NX) do buf,P,t
        mul!(buf, iRrBr(t), P)
    end

    PT,_ = arec(Ar(T), Br(T)*iRr(T)*Br(T)', Qr(T))
    Pr = solve(ODEProblem(riccati!, PT, (T,0.0), (Ar,Br,Rr,Qr,Kr)))

    Kr = Functor(NU,NX) do buf,t
        mul!(buf, iRrBr(t), Pr(t))
    end

    return Kr
end



# --------------------------------- projection --------------------------------- #



# for projection, provided Kr(t)
function stabilized_dynamics!(dx, x, (α,μ,Kr,f), t)
    u = μ(t) - Kr(t)*(x-α(t))
    dx .= f(x,u)
    # FUTURE: in-place f!(dx,x,u) 
end

# η,Kr -> ξ # projection to generate stabilized trajectory
function projection(η,Kr,model)
    @unpack model
    T = last(ts)
    (α,μ) = η

    x_ode = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (α,μ,Kr,f)))

    #TEST: performance against just returning x_ode
    x = x_ode
    # x = Functor(NX) do buf,t
    #     copy!(buf, x_ode(t))
    # end

    u = Functor(NU) do buf,t
        buf .= μ(t) - Kr(t)*(x(t)-α(t))
    end

    ξ = (x,u)
    return ξ
end


# φ,Kr -> ξ # projection
function update_ξ!(X_x,U_u,Kr,X_α,U_μ,model)
    @unpack model
    T = last(ts)
    X_ode = solve(ODEProblem(stabilized_dynamics!, x0, (0.0,T), (Kr,X_α,U_μ,model)))
    for (X, U, t) in zip(X_x, U_u, times(X_x))
        X .= X_ode(t)
        U .= U_μ(t) - Kr(t)*(X-X_α(t))
    end
end



# --------------------------------- search direction --------------------------------- #




function search_direction(ξ, η, model)
    @unpack model
    T = last(ts)
    (x,u) = ξ
    (α,μ) = η

    A = Functor(NX,NX) do buf,t
        fx!(buf, x(t), u(t))
    end

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
        mul!(buf, inv(R(t)), (S(t)'+B(t)'*P))
    end

    # --------------- solve optimizer Ko --------------- #

    PT = MArray{Tuple{NX,NX},Float64}(undef) # pxx!
    pxx!(PT, α(T)) # around unregulated trajectory

    P = solve(ODEProblem(optimizer!, PT, (T,0.0), (Ko,R,Q,A)))
    
    # Ko = inv(R)\(S'+B'*P)
    Ko = Functor(NU,NX) do buf,t
        mul!(buf, inv(R(t)), (S(t)'+B(t)'*P(t)))
    end


    # --------------- solve costate dynamics vo --------------- #

    # solve costate dynamics vo
    rT = MArray{Tuple{NX},Float64}(undef)
    px!(rT, α(T)) # around unregulated trajectory
    r = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)))



    vo = Functor(NU) do buf,t
        mul!(buf, -inv(R(t)), (B(t)'*r(t)+b(t)))
    end



    # --------------- forward integration for search direction --------------- #

    v = Functor(NU) do buf,z,t
        mul!(buf, Ko(t), z)
        buf .*= -1
        buf .+= vo(t)
    end

    
    z0 = 0 .* model.x_eq
    z_ode = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (A,B,v)))
    
    #MAYBE: interpolate? just return the ode?
    #FUTURE: find a way to return ode interpolations in-place
    z = Functor(NX) do buf,t
        copy!(buf, z_ode(t))
    end

    # v = -Ko(t)*z+vo(t)
    v = Functor(NU) do buf,t
        mul!(buf, Ko(t), z(t))
        buf .*= -1
        buf .+= vo(t)
    end

    ζ = (z,v)

    # --------------- cost derivatives --------------- #
    #FUTURE: break out to separate function
    y0 = [0;0]
    y = solve(ODEProblem(cost_derivatives!, y0, (0.0,T), (z,v,a,b,Q,S,R)))
    Dh = y(T)[1] + rT'*z(T)
    D2g = y(T)[2] + z(T)'*PT*z(T)

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


# ----------------------------------- cost derivatives ----------------------------------- #


# function cost_derivatives()
#     y0 = [0;0]
#     y = solve(ODEProblem(cost_derivatives!, y0, (0.0,T), (z,v,a,b,Qo,So,Ro)))
#     Dh = y(T)[1] + rT'*z(T)
#     D2g = y(T)[2] + z(T)'*PT*z(T)
# end



function cost_derivatives!(dy, y, (z,v,a,b,Qo,So,Ro), t)
    dy[1] = a(t)'*z(t) + b(t)'*v(t)
    dy[2] = z(t)'*Qo(t)*z(t) + 2*z(t)'*So(t)*v(t) + v(t)'*Ro(t)*v(t)
end


# ----------------------------------- armijo ----------------------------------- #

# armijo_backstep:
function armijo_backstep(x,u,Kr,z,v,Dh,model)
    γ = 1
    T = last(model.ts)
    
    # compute cost
    J = cost(x,u,model)
    h = J(T)[1] + model.p(x(T)) # around regulated trajectory
    # ξ = 0

    while γ > model.β^12
        @info "armijo update: γ = $γ"
        
        # generate estimate
        # MAYBE: α̂(γ,t)

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

        η̂ = (α̂, μ̂)
        # α = Timeseries(t->(x(t) + γ*z(t)))
        # μ = Timeseries(t->(u(t) + γ*v(t)))
        ξ̂ = projection(η̂, Kr, model)
        (x̂,û) = ξ̂

        J = cost(ξ̂..., model)
        g = J(T)[1] + model.p(x̂(T))

        # check armijo rule
        h-g >= -model.α*γ*Dh ? (return ξ̂) : (γ *= model.β)
        # println("γ=$γ, h-g=$(h-g)")
    end

    @warn "maxiters"
    return ξ̂
end



function stage_cost!(dh, h, (l,x,u), t)
    dh .= l(x(t), u(t))
end

function cost(x,u,model)
    T = last(model.t)
    h = solve(ODEProblem(stage_cost!, [0], (0.0,T), (model.l,x,u)))
    return h
end
 


# ----------------------------------- main loop ----------------------------------- #

function pronto(model)
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
    α = Interpolant(t->guess(t, model.x0, model.x_eq, T), ts, NX)
    μ = Interpolant(ts, NU)
    η = (α,μ)
    pronto(η,model)
end


function pronto(η,model)
    @info "initializing"
    ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU

    for i in 1:model.maxiters
        @info "iteration: $i"
        # η -> Kr # regulator
        tx = @elapsed begin
            Kr = regulator(η, model)
        end
        @info "(itr: $i) regulator solved in $tx seconds"

        # η,Kr -> ξ # projection
        tx = @elapsed begin
            ξ = projection(η, Kr, model)
        end
        # tx = @elapsed update_ξ!(X_x,U_u,Kr,X_α,U_μ,model)
        @info "(itr: $i) projection solved in $tx seconds"

        # φ,Kr -> ζ # search direction
        tx = @elapsed begin
            ζ,Dh = search_direction(ξ, η, model)
        end
        @info "(itr: $i) search direction found in $tx seconds"

        # # check Dh criteria -> return ξ,Kr
        @info "Dh is $Dh"
        Dh > 0 && (@warn "increased cost - quitting"; return η)
        -Dh < model.tol && (@info "PRONTO converged"; return η)
        
        # @info "calculating new trajectory:"
        # # φ,ζ,Kr -> γ -> ξ # armijo
        tx = @elapsed begin
            ξ̂ = armijo_backstep(ξ...,Kr,ζ...,Dh,model)
        end
        @info "(itr: $i) trajectory update found in $tx seconds"

        (x̂,û) = ξ̂
        update!(t->x̂(t), α)
        update!(t->û(t), μ)
        η = (α,μ)

    end
    # η is optimal (or last iteration)

    @warn "maxiters"
    return η

end





end #module