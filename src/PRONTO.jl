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
macro genfunc(fn, args)
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
            ξ = vcat(x,u)
            @variables α[1:nx(M)] 
            @variables μ[1:nu(M)] 
            @variables Pr[1:nx(M),1:nx(M)]
            @variables P[1:nx(M),1:nx(M)]
            ($fn)(M, $args...)
        end
    end
end

@genfunc f     (θ,t,x,u)
@genfunc fx    (θ,t,x,u)
@genfunc fu    (θ,t,x,u)
@genfunc fxx   (θ,t,x,u)
@genfunc fxu   (θ,t,x,u)
@genfunc fuu   (θ,t,x,u)

@genfunc l     (θ,t,x,u)
@genfunc lx    (θ,t,x,u)
@genfunc lu    (θ,t,x,u)
@genfunc lxx   (θ,t,x,u)
@genfunc lxu   (θ,t,x,u)
@genfunc luu   (θ,t,x,u)

@genfunc p     (θ,t,x,u)
@genfunc px    (θ,t,x,u)
@genfunc pxx   (θ,t,x,u)

@genfunc Rr    (θ,t,α,μ)
@genfunc Qr    (θ,t,α,μ) 
@genfunc Kr    (θ,t,α,μ,Pr)
@genfunc Ko    (θ,t,x,u,P) 

@genfunc Pr_t  (θ,t,α,μ,Pr)
@genfunc ξ_t   (θ,t,ξ,α,μ,Pr)
@genfunc P_t   (θ,t,x,u,P)


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
Ko(M::Model,θ,t,x,u,P) = throw(ModelDefError(M,:Ko))


Pr_t(M::Model,θ,t,α,μ,Pr) = throw(ModelDefError(M, :Pr_t))
ξ_t(M::Model,θ,t,ξ,α,μ,P) = throw(ModelDefError(M, :ξ_t))
P_t(M::Model,θ,t,x,u,P) = throw(ModelDefError(M, :P_t))


f!(M::Model,buf,θ,t,x,u) = throw(ModelDefError(M, :f!))
Pr_t!(M::Model,buf,θ,t,α,μ,Pr) = throw(ModelDefError(M, :Pr_t!))
ξ_t!(M::Model,buf,θ,t,ξ,α,μ,P) = throw(ModelDefError(M, :ξ_t!))
P_t!(M::Model,buf,θ,t,x,u,P) = throw(ModelDefError(M, :P_t!))


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
    
    # symbolic derivatives
    jac_sym = map(1:length(dx)) do i
        map(f_sym) do f
            derivative(f, dx[i])
        end
    end

    fx_sym = cat(jac_sym...; dims=ndims(f_sym)+1)

    if !isnothing(force_dims)
        fx_sym = reshape(fx_sym, force_dims...)
    end

    # fx_sym = cat(
    #     map(1:length(dx)) do i
    #         map(f_sym) do f
    #             derivative(f, dx[i])
    #         end
    #     end...; dims = ndims(f_sym)+1)

    # return build_function(fx_sym, args...)
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
        Rr = $(esc(:(Rr)))
        Qr = $(esc(:(Qr)))
        f = $(esc(:(f)))
        l = $(esc(:(l)))
        p = $(esc(:(p)))

        # define symbolics for derivation
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

        # derive models
        # local f,f! = build(f,θ,t,x,u) do θ,t,ξ
        #     local x,u = split(ξ)
        # end
        local f,f! = build(f,θ,t,x,u)
        local fx,fx! = Jx(f,θ,t,x,u)
        local fu,fu! = Ju(f,θ,t,x,u)
        local fxx,fxx! = Jx(fx,θ,t,x,u)
        local fxu,fxu! = Ju(fx,θ,t,x,u)
        local fuu,fuu! = Ju(fu,θ,t,x,u)

        local l,l! = build(l,θ,t,x,u)
        local lx,lx! = Jx(l,θ,t,x,u; force_dims=(nx($T()),))
        local lu,lu! = Ju(l,θ,t,x,u; force_dims=(nu($T()),))
        local lxx,lxx! = Jx(lx,θ,t,x,u)
        local lxu,lxu! = Ju(lx,θ,t,x,u)
        local luu,luu! = Ju(lu,θ,t,x,u)

        local p,p! = build(p,θ,t,x,u)
        local px,px! = Jx(p,θ,t,x,u; force_dims=(nx($T()),))
        local pxx,pxx! = Jx(px,θ,t,x,u)

        #Kr = Rr\(Br'Pr)
        local Kr,Kr! = build(θ,t,α,μ,Pr) do θ,t,α,μ,Pr
            Rr(θ,t,α,μ)\(fu(θ,t,α,μ)'*@vec(Pr))
        end

        local Pr_t,Pr_t! = build(θ,t,α,μ,Pr) do θ,t,α,μ,Pr
            riccati(fx(θ,t,α,μ), Kr(θ,t,α,μ,Pr), @vec(Pr), Qr(θ,t,α,μ), Rr(θ,t,α,μ))
        end

        local ξ_t,ξ_t! = build(θ,t,ξ,α,μ,Pr) do θ,t,ξ,α,μ,Pr
            local x,u = split($T(),ξ)
            vcat(f(θ,t,x,u)...,
                (@vec(μ) - Kr(θ,t,α,μ,Pr)*(@vec(x) - @vec(α)) - @vec(u))...)
        end
        
        #Ko = R\(S'+B'P)
        local Ko,Ko! = build(θ,t,x,u,P) do θ,t,x,u,P
            luu(θ,t,x,u)\(lxu(θ,t,x,u)' .+ fu(θ,t,x,u)'*@vec(P))
        end

        local P_t,P_t! = build(θ,t,x,u,P) do θ,t,x,u,P
            riccati(fx(θ,t,x,u), Ko(θ,t,x,u,P), @vec(P), lxx(θ,t,x,u), luu(θ,t,x,u))
        end



        # add functions to PRONTO - only at this point do we care about dispatch on the first arg
        PRONTO.Rr(M::$T,θ,t,α,μ) = Rr(θ,t,α,μ)
        PRONTO.Qr(M::$T,θ,t,α,μ) = Qr(θ,t,α,μ)
        PRONTO.Kr(M::$T,θ,t,α,μ,Pr) = Kr(θ,t,α,μ,Pr) # NU,NX
        PRONTO.Ko(M::$T,θ,t,x,u,P) = Ko(θ,t,x,u,P) # NU,NX

        PRONTO.f(M::$T,θ,t,x,u) = f(θ,t,x,u) # NX
        PRONTO.fx(M::$T,θ,t,x,u) = fx(θ,t,x,u) # NX,NX
        PRONTO.fu(M::$T,θ,t,x,u) = fu(θ,t,x,u) # NX,NU
        PRONTO.fxx(M::$T,θ,t,x,u) = fxx(θ,t,x,u) # NX,NX,NX
        PRONTO.fxu(M::$T,θ,t,x,u) = fxu(θ,t,x,u) # NX,NX,NU
        PRONTO.fuu(M::$T,θ,t,x,u) = fuu(θ,t,x,u) # NX,NU,NU

        PRONTO.l(M::$T,θ,t,x,u) = l(θ,t,x,u) # 1
        PRONTO.lx(M::$T,θ,t,x,u) = lx(θ,t,x,u) # NX
        PRONTO.lu(M::$T,θ,t,x,u) = lu(θ,t,x,u) # NU
        PRONTO.lxx(M::$T,θ,t,x,u) = lxx(θ,t,x,u) # NX,NX
        PRONTO.lxu(M::$T,θ,t,x,u) = lxu(θ,t,x,u) # NX,NU
        PRONTO.luu(M::$T,θ,t,x,u) = luu(θ,t,x,u) # NU,NU

        PRONTO.p(M::$T,θ,t,x,u) = p(θ,t,x,u) # 1
        PRONTO.px(M::$T,θ,t,x,u) = px(θ,t,x,u) # NX
        PRONTO.pxx(M::$T,θ,t,x,u) = pxx(θ,t,x,u) # NX,NX

        PRONTO.Pr_t(M::$T,θ,t,α,μ,Pr) = Pr_t(θ,t,α,μ,Pr) # NX,NX
        PRONTO.ξ_t(M::$T,θ,t,ξ,α,μ,P) = ξ_t(θ,t,ξ,α,μ,P) # NX+NU
        PRONTO.P_t(M::$T,θ,t,x,u,P) = P_t(θ,t,x,u,P) # NX,NX

        PRONTO.f!(M::$T,buf,θ,t,x,u) = f!(buf,θ,t,x,u) #NX
        PRONTO.Pr_t!(M::$T,buf,θ,t,α,μ,Pr) = Pr_t!(buf,θ,t,α,μ,Pr) # NX,NX    
        PRONTO.ξ_t!(M::$T,buf,θ,t,ξ,α,μ,P) = ξ_t!(buf,θ,t,ξ,α,μ,P) # NX+NU
        PRONTO.P_t!(M::$T,buf,θ,t,x,u,P) = P_t!(buf,θ,t,x,u,P) # NX,NX    

        info("$($T) model derivation complete!")
    end
end


# ----------------------------------- ode solution handling ----------------------------------- #
include("utils.jl")

# ----------------------------------- main loop ----------------------------------- #

Pr_ode(dPr, Pr, (M,φ,θ), t) = Pr_t!(M, dPr, θ, t, φ(t)..., Pr)
ξ_ode(dξ, ξ, (M,θ,φ,Pr), t) = ξ_t!(M, dξ, θ, t, ξ, φ(t)..., Pr(t))
P_ode(dP, P, (M,ξ,θ), t) = P_t!(M, dP, θ, t, ξ(t)..., P)



function pronto(M::Model, θ, t0, tf, x0, u0, φ)
    
    info("solving regulator")
    Prf = diagm(ones(nx(M)))
    Pr = Solution(ODEProblem(Pr_ode,Prf,(tf,t0),(M,φ,θ)), nx(M), nx(M))

    info("solving projection")
    ξ = Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))

    info("solving optimizer")
    Pf = pxx(M,θ,tf,φ(tf)...)
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