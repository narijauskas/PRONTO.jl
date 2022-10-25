# PRONTO.jl v0.3.0-dev
module PRONTO
# include("kernels.jl")
using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
using LinearAlgebra
using UnicodePlots
# default_size!(;width=80)

using DifferentialEquations
using Symbolics
using Symbolics: derivative

export @derive
export pronto
export nx,nu,nθ

export Buffer
export Solution
export Trajectory
export preview

# ----------------------------------- base definitions ----------------------------------- #


abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ


struct ModelDefError <: Exception
    M::Model
    fxn::Symbol
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.M)
    print(io, 
        "PRONTO.$(e.fxn) is missing a method for the $T model.\n",
        "Please check the $T model definition and then re-run: ",
        "@derive $T\n"
    )
end


# M not included in args
macro genfunc(fn, args...)
    fn = esc(fn)
    return quote
        # define a default fallback method
        # function ($fn)(M::Model, $(args...))
        #     throw(ModelDefError(M,nameof($fn)))
        # end
        # define a symbolic rendition
        function ($fn)(M::Model)
            @variables θ[1:nθ(M)]
            @variables t
            @variables x[1:nx(M)] 
            @variables u[1:nu(M)]
            @variables α[1:nx(M)] 
            @variables μ[1:nu(M)] 
            @variables Pr[1:nx(M),1:nx(M)]
            ($fn)($args...)
        end
    end
end

@genfunc f     (θ,t,x,u)
@genfunc fx    (θ,t,x,u)
@genfunc fu    (θ,t,x,u)
@genfunc fxx   (θ,t,x,u)
@genfunc fxu   (θ,t,x,u)
@genfunc fuu   (θ,t,x,u)
@genfunc Rr    (θ,t,α,μ) 
@genfunc Qr    (θ,t,α,μ) 
@genfunc Kr    (θ,t,α,μ,Pr) 
@genfunc Pr_t   (θ,t,α,μ,Pr)
@genfunc ξ_t    (θ,t,x,u,α,μ,Pr)


f(M::Model,θ,t,x,u) = throw(ModelDefError(M,:f))
fx(M::Model,θ,t,x,u) = throw(ModelDefError(M,:fx))
fu(M::Model,θ,t,x,u) = throw(ModelDefError(M,:fu))
fxx(M::Model,θ,t,x,u) = throw(ModelDefError(M,:fxx))
fxu(M::Model,θ,t,x,u) = throw(ModelDefError(M,:fxu))
fuu(M::Model,θ,t,x,u) = throw(ModelDefError(M,:fuu))

l(M::Model,θ,t,x,u) = throw(ModelDefError(M,:l))
lx(M::Model,θ,t,x,u) = throw(ModelDefError(M,:lx))
lu(M::Model,θ,t,x,u) = throw(ModelDefError(M,:lu))
lxx(M::Model,θ,t,x,u) = throw(ModelDefError(M,:lxx))
lxu(M::Model,θ,t,x,u) = throw(ModelDefError(M,:lxu))
luu(M::Model,θ,t,x,u) = throw(ModelDefError(M,:luu))

p(M::Model,θ,t,x,u) = throw(ModelDefError(M,:p))
px(M::Model,θ,t,x,u) = throw(ModelDefError(M,:px))
pxx(M::Model,θ,t,x,u) = throw(ModelDefError(M,:pxx))

Rr(M::Model,θ,t,α,μ) = throw(ModelDefError(M,:Rr))
Qr(M::Model,θ,t,α,μ) = throw(ModelDefError(M,:Qr))
Kr(M::Model,θ,t,α,μ,P) = throw(ModelDefError(M,:Kr))

Pr_t(M::Model,θ,t,α,μ,Pr) = throw(ModelDefError(M, :Pr_t))
ξ_t(M::Model,θ,t,x,u,α,μ,P) = throw(ModelDefError(M, :ξ_t))


f!(M::Model,buf,θ,t,x,u) = throw(ModelDefError(M, :f!))
Pr_t!(M::Model,buf,θ,t,α,μ,Pr) = throw(ModelDefError(M, :Pr_t!))
ξ_t!(M::Model,buf,θ,t,x,u,α,μ,P) = throw(ModelDefError(M, :ξ_t!))


# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(M,buf,...) = @error
# - symbolic generator symbolic(M,f)


riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q



# @genfunc(:Kr,θ,t,α,μ,P)
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
        @variables θ[1:nθ($T())]
        @variables t
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())]
        @variables α[1:nx($T())] 
        @variables μ[1:nu($T())] 
        @variables Pr[1:nx($T()),1:nx($T())]

        Jx,Ju = Jacobian.([x,u])

        # derive models
        local f,f! = build(f,θ,t,x,u)
        local fx,fx! = Jx(f,θ,t,x,u)
        local fu,fu! = Ju(f,θ,t,x,u)
        local fxx,fxx! = Jx(fx,θ,t,x,u)

        local Kr,Kr! = build(θ,t,α,μ,Pr) do θ,t,α,μ,Pr
            Rr(θ,t,α,μ)\(fu(θ,t,α,μ)'*collect(Pr))
        end

        local Pr_t,Pr_t! = build(θ,t,α,μ,Pr) do θ,t,α,μ,Pr
            riccati(fx(θ,t,α,μ), Kr(θ,t,α,μ,Pr), collect(Pr), Qr(θ,t,α,μ), Rr(θ,t,α,μ))
        end

        local ξ_t,ξ_t! = build(θ,t,x,u,α,μ,Pr) do θ,t,x,u,α,μ,Pr
            vcat(
                f(θ,t,x,u)...,
                (collect(μ) - Kr(θ,t,α,μ,Pr)*(collect(x)-collect(α)) - collect(u))...
            )
        end




        # add functions to PRONTO - only at this point do we care about dispatch on the first arg
        PRONTO.Rr(M::$T,θ,t,α,μ) = Rr(θ,t,α,μ)
        PRONTO.Qr(M::$T,θ,t,α,μ) = Qr(θ,t,α,μ)
        PRONTO.Kr(M::$T,θ,t,α,μ,P) = Kr(θ,t,α,μ,P) # NU,NX

        PRONTO.f(M::$T,θ,t,x,u) = f(θ,t,x,u) # NX
        PRONTO.fx(M::$T,θ,t,x,u) = fx(θ,t,x,u) # NX,NX
        PRONTO.fu(M::$T,θ,t,x,u) = fu(θ,t,x,u) # NX,NU
        PRONTO.fxx(M::$T,θ,t,x,u) = fxx(θ,t,x,u) # NX,NX
        
        PRONTO.Pr_t(M::$T,θ,t,α,μ,Pr) = Pr_t(θ,t,α,μ,Pr) # NX,NX
        PRONTO.ξ_t(M::$T,θ,t,x,u,α,μ,P) = ξ_t(θ,t,x,u,α,μ,P) # NX+NU
        
        
        PRONTO.f!(M::$T,buf,θ,t,x,u) = f!(buf,θ,t,x,u) #NX
        PRONTO.Pr_t!(M::$T,buf,θ,t,α,μ,Pr) = Pr_t!(buf,θ,t,α,μ,Pr) # NX,NX    
        PRONTO.ξ_t!(M::$T,buf,θ,t,x,u,α,μ,P) = ξ_t!(buf,θ,t,x,u,α,μ,P) # NX+NU

        @info "$($T) model derivation complete!"
    end
end


# ----------------------------------- ode solution handling ----------------------------------- #
include("utils.jl")

# ----------------------------------- main ----------------------------------- #
# where φ=(α,μ)::Trajectory{NX,NU}
function Pr_ode(dPr, Pr,(M,φ,θ), t)
    Pr_t!(M,dPr,θ,t,φ(t)...,Pr)
end


function ξ_ode(dξ,ξ,(M,θ,φ,Pr),t)
    x = @view ξ[1:nx(M)]
    u = @view ξ[nx(M)+1:end]
    ξ_t!(M,dξ,θ,t,x,u,φ(t)...,Pr(t))
end


function pronto(M::Model, θ, t0, tf, x0, u0, φ)
    Prf = diagm(ones(nx(M)))
    
    @info "solving regulator"
    Pr = Solution(ODEProblem(Pr_ode,Prf,(t0,tf),(M,φ,θ)), nx(M), nx(M))
    @info "solving projection"
    ξ = Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))
end

# Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))



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

# include("utils.jl")
include("guess.jl")

end # module