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
export info
export nx,nu,nθ

export Buffer
export Solution
export Trajectory
export preview




# ----------------------------------- runtime feedback ----------------------------------- #
using Crayons
as_tag(str) = as_tag(crayon"default", str)
as_tag(c::Crayon, str) = as_color(c, as_bold("[$str: "))
as_color(c::Crayon, str) = "$c" * str * "$(crayon"default")"
as_bold(str) = "$(crayon"bold")" * str * "$(crayon"!bold")"
clearln() = print("\e[2K","\e[1G")

info(str) = println(as_tag(crayon"magenta","PRONTO"), str)
info(i, str) = println(as_tag(crayon"magenta","PRONTO[$i]"), str)
# tinfo(i, str, tx) = println(as_tag(crayon"magenta","PRONTO[$i]"), str, " in $(round(tx*1000; digits=2)) ms")



# ----------------------------------- helper functions ----------------------------------- #
# mapid!(dest, src) = map!(identity, dest, src)
# same as: mapid!(dest, src) = map!(x->x, dest, src)

inv!(A) = LinearAlgebra.inv!(lu!(A)) # general
# LinearAlgebra.inv!(choelsky!(A)) # if SPD



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
# @fndef
# @argdef
macro genfunc(fn, args)
    fn = esc(fn)
    return quote
        # define a default fallback method
        function ($fn)(M::Model, $(args.args...))
            throw(ModelDefError(M,nameof($fn)))
        end
        # define a symbolic rendition
        function ($fn)(M::Model)
            @variables θ[1:nθ(M)]
            @variables t
            @variables x[1:nx(M)] 
            @variables u[1:nu(M)]
            ξ = vcat(x,u)
            @variables α[1:nx(M)] 
            @variables μ[1:nu(M)] 
            @variables Pr[1:nx(M),1:nx(M)]
            @variables P[1:nx(M),1:nx(M)]
            ($fn)(M, $args...)
        end
    end
end

@genfunc f     (θ,t,ξ)
@genfunc fx    (θ,t,ξ)
@genfunc fu    (θ,t,ξ)
@genfunc fxx   (θ,t,ξ)
@genfunc fxu   (θ,t,ξ)
@genfunc fuu   (θ,t,ξ)

@genfunc l     (θ,t,ξ)
@genfunc lx    (θ,t,ξ)
@genfunc lu    (θ,t,ξ)
@genfunc lxx   (θ,t,ξ)
@genfunc lxu   (θ,t,ξ)
@genfunc luu   (θ,t,ξ)

@genfunc p     (θ,t,ξ)
@genfunc px    (θ,t,ξ)
@genfunc pxx   (θ,t,ξ)

@genfunc Rr    (θ,t,φ)
@genfunc Qr    (θ,t,φ) 
@genfunc Kr    (θ,t,φ,Pr)
@genfunc Ko    (θ,t,ξ,P) 

@genfunc Pr_t  (θ,t,φ,Pr)
@genfunc ξ_t   (θ,t,ξ,φ,Pr)
@genfunc P_t   (θ,t,ξ,P)


f!(M::Model,buf,θ,t,ξ) = throw(ModelDefError(M, :f!))
Pr_t!(M::Model,buf,θ,t,φ,Pr) = throw(ModelDefError(M, :Pr_t!))
ξ_t!(M::Model,buf,θ,t,ξ,φ,P) = throw(ModelDefError(M, :ξ_t!))
P_t!(M::Model,buf,θ,t,ξ,P) = throw(ModelDefError(M, :P_t!))


# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(M,buf,...) = @error
# - symbolic generator symbolic(M,f)


riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q



# @genfunc(:Kr,θ,t,α,μ,P)
# ----------------------------------- symbolics & autodiff ----------------------------------- #



function build(f, args...)
    f_sym = collect(Base.invokelatest(f, args...))
    f_ex = build_function(f_sym, args...)
    return eval.(f_ex)
end

function jacobian(dx, f, args...; force_dims=nothing)
    f_sym = collect(Base.invokelatest(f, args...))
    
    # generate symbolic derivatives
    jac_sym = map(1:length(dx)) do i
        map(f_sym) do f
            derivative(f, dx[i])
        end
    end

    # reshape/concatenate
    fx_sym = cat(jac_sym...; dims=ndims(f_sym)+1)
    if !isnothing(force_dims)
        fx_sym = reshape(fx_sym, force_dims...)
    end

    fx_ex = build_function(fx_sym, args...)
    return eval.(fx_ex)
end

struct Jacobian
    dx
end
(J::Jacobian)(f, args...; kw...) = jacobian(J.dx, f, args...; kw...)


# ----------------------------------- model derivation ----------------------------------- #

# vectorization helper for autodiff -> wrap symbolic array variables
# really, just shorter than typing collect :P
macro vec(ex)
    :(collect($ex))
end

split(M::Model, ξ) = (ξ[1:nx(M)], ξ[(nx(M)+1):end])

#YO: #FUTURE: if pronto knows each function's argument signature
# can it create a representation that expands them, eg. @Rr->Rr(θ,t,α,μ) in the @derive macro

# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro derive(T)
    # make sure we use the local context
    T = esc(T)
    return quote
        info("starting $($T) model derivation")
        # load user definitions
        # Rr = $(esc(:(Rr)))
        # Qr = $(esc(:(Qr)))
        # f = $(esc(:(f)))
        # l = $(esc(:(l)))
        # p = $(esc(:(p)))

        println("\t > defining symbolic variables and operators")
        #MAYBE: pre-collect so user doesn't have to?
        @variables θ[1:nθ($T())]
        @variables t[1:1]
        @variables x[1:nx($T())]
        @variables u[1:nu($T())]
        ξ = vcat(x,u)
        @variables α[1:nx($T())] 
        @variables μ[1:nu($T())] 
        φ = vcat(α,μ)
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables P[1:nx($T()),1:nx($T())]

        Jx,Ju = Jacobian.([x,u])

        println("\t > processing regulator equations")
        
        # load user function, remap ξ<->(x,u)
        local Qr,Qr! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(Qr))))(θ,t,x,u)
        end

        local Rr,Rr! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(Rr))))(θ,t,x,u)
        end

        #add definitions to PRONTO
        PRONTO.Rr(M::$T,θ,t,φ) = Rr(θ,t,φ)
        PRONTO.Qr(M::$T,θ,t,φ) = Qr(θ,t,φ)
        println("\t > differentiating dynamics f")

        # load user function, remap ξ<->(x,u)
        local f,f! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(f))))(θ,t,x,u)
        end

        # derive models
        local fx,fx! = Jx(f,θ,t,ξ)
        local fu,fu! = Ju(f,θ,t,ξ)
        local fxx,fxx! = Jx(fx,θ,t,ξ)
        local fxu,fxu! = Ju(fx,θ,t,ξ)
        local fuu,fuu! = Ju(fu,θ,t,ξ)

        # add definitions to PRONTO
        PRONTO.f(M::$T,θ,t,ξ) = f(θ,t,ξ) # NX
        PRONTO.fx(M::$T,θ,t,ξ) = fx(θ,t,ξ) # NX,NX
        PRONTO.fu(M::$T,θ,t,ξ) = fu(θ,t,ξ) # NX,NU
        PRONTO.fxx(M::$T,θ,t,ξ) = fxx(θ,t,ξ) # NX,NX,NX
        PRONTO.fxu(M::$T,θ,t,ξ) = fxu(θ,t,ξ) # NX,NX,NU
        PRONTO.fuu(M::$T,θ,t,ξ) = fuu(θ,t,ξ) # NX,NU,NU
        PRONTO.f!(M::$T,buf,θ,t,ξ) = f!(buf,θ,t,ξ) #NX

        println("\t > differentiating stage cost l")

        # load user function, remap ξ<->(x,u)
        local l,l! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(l))))(θ,t,x,u)
        end

        # derive models
        local lx,lx! = Jx(l,θ,t,ξ; force_dims=(nx($T()),))
        local lu,lu! = Ju(l,θ,t,ξ; force_dims=(nu($T()),))
        local lxx,lxx! = Jx(lx,θ,t,ξ)
        local lxu,lxu! = Ju(lx,θ,t,ξ)
        local luu,luu! = Ju(lu,θ,t,ξ)

        PRONTO.l(M::$T,θ,t,ξ) = l(θ,t,ξ) # 1
        PRONTO.lx(M::$T,θ,t,ξ) = lx(θ,t,ξ) # NX
        PRONTO.lu(M::$T,θ,t,ξ) = lu(θ,t,ξ) # NU
        PRONTO.lxx(M::$T,θ,t,ξ) = lxx(θ,t,ξ) # NX,NX
        PRONTO.lxu(M::$T,θ,t,ξ) = lxu(θ,t,ξ) # NX,NU
        PRONTO.luu(M::$T,θ,t,ξ) = luu(θ,t,ξ) # NU,NU

        println("\t > differentiating terminal cost p")

        # load user function, remap ξ<->(x,u)
        local p,p! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(p))))(θ,t,x,u)
        end

        # derive models
        local px,px! = Jx(p,θ,t,ξ; force_dims=(nx($T()),))
        local pxx,pxx! = Jx(px,θ,t,ξ)

        PRONTO.p(M::$T,θ,t,ξ) = p(θ,t,ξ) # 1
        PRONTO.px(M::$T,θ,t,ξ) = px(θ,t,ξ) # NX
        PRONTO.pxx(M::$T,θ,t,ξ) = pxx(θ,t,ξ) # NX,NX

        println("\t > deriving regulator equations")

        #Kr = Rr\(Br'Pr)
        local Kr,Kr! = build(θ,t,φ,Pr) do θ,t,φ,Pr
            Rr(θ,t,φ)\(fu(θ,t,φ)'*@vec(Pr))
        end

        local Pr_t,Pr_t! = build(θ,t,φ,Pr) do θ,t,φ,Pr
            riccati(fx(θ,t,φ), Kr(θ,t,φ,Pr), @vec(Pr), Qr(θ,t,φ), Rr(θ,t,φ))
        end

        PRONTO.Kr(M::$T,θ,t,φ,Pr) = Kr(θ,t,φ,Pr) # NU,NX
        PRONTO.Pr_t(M::$T,θ,t,φ,Pr) = Pr_t(θ,t,φ,Pr) # NX,NX
        PRONTO.Pr_t!(M::$T,buf,θ,t,φ,Pr) = Pr_t!(buf,θ,t,φ,Pr) # NX,NX    

        println("\t > deriving projection equations")

        local ξ_t,ξ_t! = build(θ,t,ξ,φ,Pr) do θ,t,ξ,φ,Pr
            local x,u = split($T(),ξ)
            local α,μ = split($T(),φ)

            vcat(f(θ,t,ξ)...,
                (@vec(μ) - Kr(θ,t,φ,Pr)*(@vec(x) - @vec(α)) - @vec(u))...)
        end

        PRONTO.ξ_t(M::$T,θ,t,ξ,φ,P) = ξ_t(θ,t,ξ,φ,P) # NX+NU
        PRONTO.ξ_t!(M::$T,buf,θ,t,ξ,φ,P) = ξ_t!(buf,θ,t,ξ,φ,P) # NX+NU
        
        println("\t > deriving optimizer equations")

        #Ko = R\(S'+B'P)
        local Ko,Ko! = build(θ,t,ξ,P) do θ,t,ξ,P
            luu(θ,t,ξ)\(lxu(θ,t,ξ)' .+ fu(θ,t,ξ)'*@vec(P))
        end

        local P_t,P_t! = build(θ,t,ξ,P) do θ,t,ξ,P
            riccati(fx(θ,t,ξ), Ko(θ,t,ξ,P), @vec(P), lxx(θ,t,ξ), luu(θ,t,ξ))
        end

        PRONTO.Ko(M::$T,θ,t,ξ,P) = Ko(θ,t,ξ,P) # NU,NX
        PRONTO.P_t(M::$T,θ,t,ξ,P) = P_t(θ,t,ξ,P) # NX,NX
        PRONTO.P_t!(M::$T,buf,θ,t,ξ,P) = P_t!(buf,θ,t,ξ,P) # NX,NX    

        info("$($T) model derivation complete!")
    end
end


# ----------------------------------- ode solution handling ----------------------------------- #
include("utils.jl")

# ----------------------------------- main loop ----------------------------------- #

Pr_ode(dPr, Pr, (M,φ,θ), t) = Pr_t!(M, dPr, θ, t, φ(t), Pr)
ξ_ode(dξ, ξ, (M,θ,φ,Pr), t) = ξ_t!(M, dξ, θ, t, ξ, φ(t), Pr(t))
P_ode(dP, P, (M,ξ,θ), t) = P_t!(M, dP, θ, t, ξ(t), P)


function pronto(M::Model, θ, t0, tf, x0, u0, φ)
    
    info("solving regulator")
    Prf = diagm(ones(nx(M)))
    Pr = Solution(ODEProblem(Pr_ode,Prf,(tf,t0),(M,φ,θ)), nx(M), nx(M))
    # Pr = Solution(M, Pr_ode, Prf, (t0,tf), (M,φ,θ))

    info("solving projection")
    ξ = Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))

    info("solving optimizer")
    Pf = pxx(M,θ,tf,φ(tf))
    P = Solution(ODEProblem(P_ode,Pf,(tf,t0),(M,ξ,θ)), nx(M), nx(M))
    return ξ
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