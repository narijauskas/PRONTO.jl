# PRONTO.jl v0.3.0-dev
module PRONTO
# include("kernels.jl")
using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures

using DifferentialEquations
using Symbolics
using Symbolics: derivative

export Buffer, @buffer
export @derive
export nx,nu,nθ
export Solution



# ----------------------------------- symbolics & autodiff ----------------------------------- #

function build(f, args...)
    f_sym = cat(Base.invokelatest(f, args...); dims=1)
    f_ex = build_function(f_sym, args...)
    return eval.(f_ex)
end

function jacobian(dx, f, args...)
    f_sym = Base.invokelatest(f, args...)

    # symbolic derivatives
    fx_sym = cat(
        map(1:length(dx)) do i
            map(f_sym) do f
                derivative(f, dx[i])
            end
        end...; dims = ndims(f_sym)+1)

    # return build_function(fx_sym, args...)
    fx_ex = build_function(fx_sym, args...)
    return eval.(fx_ex)
end


struct Jacobian
    dx
end
(J::Jacobian)(f, args...) = jacobian(J.dx, f, args...)
  

# ----------------------------------- model derivation ----------------------------------- #


abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ

f(M,x,u,t,θ) = @error "PRONTO.f is missing a method for the $(typeof(M)) model."
fx(M,x,u,t,θ) = @error "PRONTO.fx is missing a method for the $(typeof(M)) model."
fu(M,x,u,t,θ) = @error "PRONTO.fu is missing a method for the $(typeof(M)) model."
fxx(M,x,u,t,θ) = @error "PRONTO.fxx is missing a method for the $(typeof(M)) model."

# l(M,x,u,t,θ)
# p(M,x,u,t,θ)

Rr(M,x,u,t,θ) = @error "PRONTO.Rr is missing a method for the $(typeof(M)) model."
Qr(M,x,u,t,θ) = @error "PRONTO.Qr is missing a method for the $(typeof(M)) model."
Kr(M,x,u,t,θ,P) = @error "PRONTO.Kr is missing a method for the $(typeof(M)) model."


f!(buf,M,x,u,t,θ) = @error "PRONTO.f! is missing a method for the $(typeof(M)) model."
Kr!(buf,M,x,u,t,θ,P) = @error "PRONTO.Kr! is missing a method for the $(typeof(M)) model."
dPr!(M,buf,x,u,t,θ,P) = nothing
dPr(M,x,u,t,θ,P) = nothing

#TODO: Kr
#TODO: Pt

# ModelDefinitionError
# $(typeof(M)) is missing a method definition for "PRONTO.fx"
# "ensure `f(...)` is correctly defined and then run `@configure T`"
# need to know: model type T, function name (eg. fx), function origin (eg. f)


# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(buf,M,...) = @error
# - symbolic generator symbolic(M,f)


riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q


# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro derive(T)
    T = esc(T) # make sure we use the local context
    return quote
        @info "starting $($T) model derivation"
        # load user definitions
        local Rr = $(esc(:(Rr)))
        local Qr = $(esc(:(Qr)))
        local f = $(esc(:(f)))
        local l = $(esc(:(l)))
        local p = $(esc(:(p)))

        # define symbolics for derivation
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())] 
        @variables t
        @variables θ[1:nθ($T())]
        @variables Pr[1:nx($T()),1:nx($T())]
        Jx,Ju = Jacobian.([x,u])

        # derive models
        local f,f! = build(f,x,u,t,θ)
        local fx,fx! = Jx(f,x,u,t,θ)
        local fu,fu! = Ju(f,x,u,t,θ)
        local fxx,fxx! = Jx(fx,x,u,t,θ)

        # local fx = allocating(jacobian(x,f,x,u,t,θ; inplace=false))
        # local fx = build(Jx(f,x,u,t,θ))
        
        # local _Kr = (x,u,t,θ,P) -> (Rr(x,u,t,θ)\(fu(x,u,t,θ)'*collect(P)))
        # local Kr,Kr! = build(_Kr,x,u,t,θ,P)
        local Kr,Kr! = build(x,u,t,θ,Pr) do x,u,t,θ,Pr
            Rr(x,u,t,θ)\(fu(x,u,t,θ)'*collect(Pr))
        end


        local dPr,dPr! = build(x,u,t,θ,Pr) do x,u,t,θ,Pr
            riccati(fx(x,u,t,θ), Kr(x,u,t,θ,Pr), collect(Pr), Qr(x,u,t,θ), Rr(x,u,t,θ))
        end

            # -fx(x,u,t,θ)'*collect(P) - collect(P)*fx(x,u,t,θ) + Kr(x,u,t,θ,P)'*Rr(x,u,t,θ)*Kr(x,u,t,θ,P) - Qr(x,u,t,θ)
        # dPr! =  
        # dP .= -Ar(t)'*P - P*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
        # local Kr = inplace(x,u,t,θ,P; inplace=false) do (x,u,t,θ,P)
        #     Rr(x,u,t,θ)\(fu(x,u,t,θ)'P)
        # end


        # add functions to PRONTO - only at this point do we care about dispatch on the first arg
        PRONTO.Rr(M::$T,x,u,t,θ) = Rr(x,u,t,θ)
        PRONTO.Qr(M::$T,x,u,t,θ) = Qr(x,u,t,θ)
        PRONTO.f(M::$T,x,u,t,θ) = f(x,u,t,θ) # NX
        PRONTO.f!(buf,M::$T,x,u,t,θ) = f!(buf,x,u,t,θ)
        # PRONTO.f!(buf,M::$T,x,u,t,θ) = (f!(buf,x,u,t,θ); return buf) # NX
        PRONTO.fx(M::$T,x,u,t,θ) = fx(x,u,t,θ) # NX,NX
        PRONTO.fu(M::$T,x,u,t,θ) = fu(x,u,t,θ) # NX,NU
        PRONTO.fxx(M::$T,x,u,t,θ) = fxx(x,u,t,θ) # NX,NX
        PRONTO.Kr(M::$T,x,u,t,θ,P) = Kr(x,u,t,θ,P) # NU,NX
        PRONTO.Kr!(buf,M::$T,x,u,t,θ,P) = Kr!(buf,x,u,t,θ,P) # NU,NX
        PRONTO.dPr!(M::$T,buf,x,u,t,θ,P) = dPr!(buf,x,u,t,θ,P) # NX,NX
        PRONTO.dPr(M::$T,x,u,t,θ,P) = dPr(x,u,t,θ,P) # NX,NX

        @info "$($T) model derivation complete!"
    end
end

# where φ=(α,μ)::Trajectory{NX,NU}
function Pr_ode(dPr, Pr,(M,φ,θ), t)
    dPr!(M,dPr,φ(t)...,t,θ,Pr)
end

# ----------------------------------- ode solution handling ----------------------------------- #


#MArray{S,T,N,L}
Buffer(S...) = MArray{Tuple{S...}, Float64, length(S), prod(S)}

macro buffer(S,f)
    # S = esc(S)
    # f = esc(f)
    return quote
        FunctionWrapper{Buffer($S...), Tuple{Float64}}(@closure t->Buffer($S...)($f(t)))
    end
end

# maps t->v::T
struct Solution{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    sln::SciMLBase.AbstractODESolution
end

(sln::Solution)(t) = sln.fxn(t)

# T = BufferType(S...)
function Solution(prob, T)
    sln = solve(prob)
    buf = T(undef)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->sln(buf, t))
    Solution(fxn,buf,sln)
end





# # maps t->ξ=(x,u)::(TX,TU)
# struct Trajectory{TX,TU}
#     x::FunctionWrapper{TX, Tuple{Float64}}
#     u::FunctionWrapper{TU, Tuple{Float64}}
#     xbuf::TX
#     ubuf::TU
#     sln::SciMLBase.AbstractODESolution
# end
# (ξ::Trajectory)(t) = (ξ.x(t), ξ.u(x,t))


# function Trajectory(prob, TX, ctrl, TU)
#     sln = solve(prob) # solves for x(t)
#     xbuf = TX(undef)
#     ubuf = TU(undef)
#     x = FunctionWrapper{TX, Tuple{Float64}}(t->sln(buf,t))
#     u = FunctionWrapper{TU, Tuple{Float64}}(t->ctrl(buf,t))
# end


# u(x,t)
# u = μ(t) - Kr(α,μ,t,θ,Pr)*(x(t)-α(t))




Base.show(io::IO, sln::Solution) = show(io,typeof(sln))
#FUTURE: show size, length, time span, solver method?

# this might be type piracy... but prevents the obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper)
    print(io, "$(typeof(fn)) $(fn.ptr)")
end







# ----------------------------------- main ----------------------------------- #

# Pr = Solution(ODEProblem(Pr_ode, zeros(2), (0.0,2.0),(M,φ,θ)))

# function Pr_ode(dPr, Pr, (?), t)
#     dPr!(dP,α,μ,t,θ)
# end


# Kr(t,P) = ... 



# function pronto(θ,x0,t0,tf,αg,μg)
 
#     Pr = solution(ODEProblem)

# end
# fx(θ,x,u,t)'*P - P*fx(θ,x,u,t)

# function riccati!(dP, P, (Ar,Br,Rr,Qr), t)
#     # mul!(Kr, Rr(t)\Br(t)', P)
#     # Kr = Rr(t)\Br(t)'*P
#     dP .= -Ar(t)'*P - P*Ar(t) + Kr(t,P)'*Rr(t)*Kr(t,P) - Qr(t)
#     #NOTE: dP is symmetric, as should be P
# end


# # pronto(θ::Kernel, x0, T/dt, θ, xg, ug)
# # pronto(M, x0, T/dt, θ, guess(...)...)
# function pronto(θ::Kernel{NX,NU},t,args...) where {NX,NU}
#     f(M,x,u,t)
# end
# # fallback: if type is given, creates an instance
# pronto(T::DataType, args...) = pronto(T(), args...)

include("utils.jl")


end # module