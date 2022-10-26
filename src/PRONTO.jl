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



inv!(A) = LinearAlgebra.inv!(lu!(A)) # general
# LinearAlgebra.inv!(choelsky!(A)) # if SPD



riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q
costate(A,B,a,b,K,x) = -(A-B*K)'x - a + K'b



# ----------------------------------- model definitions ----------------------------------- #

abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ

# auto-defines some useful methods for core model functions
# a good place to look for a complete list of PRONTO's generated internal functions
include("kernel.jl")


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
    # M = T()
    user_Qr = esc(:Qr)

    return quote
        println()
        info("deriving the $(as_bold("$($T)")) model:")
        _t0_derivation = time_ns()
        # define symbolic variables and operators
        @variables θ[1:nθ($T())]
        @variables t
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())]
        ξ = vcat(x,u)
        @variables α[1:nx($T())] 
        @variables μ[1:nu($T())]
        φ = vcat(α,μ)
        @variables z[1:nx($T())] 
        @variables v[1:nu($T())]
        ζ = vcat(z,v)
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables Po[1:nx($T()),1:nx($T())]
        @variables ro[1:nx($T())]
        @variables λ[1:nx($T())]

        Jx,Ju = Jacobian.([x,u])

        println("\t > regulator equations")
        
        # load user function, remap ξ<->(x,u)
        local Qr,Qr! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            # ($user_Qr)(θ,t,x,u)
            ($(esc(:(Qr))))(θ,t,x,u)

        end

        local Rr,Rr! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(Rr))))(θ,t,x,u)
        end

        #add definitions to PRONTO
        PRONTO.Rr(M::$T,θ,t,φ) = Rr(θ,t,φ)
        PRONTO.Qr(M::$T,θ,t,φ) = Qr(θ,t,φ)
        println("\t > dynamics derivatives")

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

        println("\t > stage cost derivatives")

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

        # add definitions to PRONTO
        PRONTO.l(M::$T,θ,t,ξ) = l(θ,t,ξ) # 1
        PRONTO.lx(M::$T,θ,t,ξ) = lx(θ,t,ξ) # NX
        PRONTO.lu(M::$T,θ,t,ξ) = lu(θ,t,ξ) # NU
        PRONTO.lxx(M::$T,θ,t,ξ) = lxx(θ,t,ξ) # NX,NX
        PRONTO.lxu(M::$T,θ,t,ξ) = lxu(θ,t,ξ) # NX,NU
        PRONTO.luu(M::$T,θ,t,ξ) = luu(θ,t,ξ) # NU,NU

        println("\t > terminal cost derivatives")

        # load user function, remap ξ<->(x,u)
        local p,p! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(p))))(θ,t,x,u)
        end

        # derive models
        local px,px! = Jx(p,θ,t,ξ; force_dims=(nx($T()),))
        local pxx,pxx! = Jx(px,θ,t,ξ)

        # add definitions to PRONTO
        PRONTO.p(M::$T,θ,t,ξ) = p(θ,t,ξ) # 1
        PRONTO.px(M::$T,θ,t,ξ) = px(θ,t,ξ) # NX
        PRONTO.pxx(M::$T,θ,t,ξ) = pxx(θ,t,ξ) # NX,NX

        println("\t > regulator solver")

        #Kr = Rr\(Br'Pr)
        local Kr,Kr! = build(θ,t,φ,Pr) do θ,t,φ,Pr
            Rr(θ,t,φ)\(fu(θ,t,φ)'*@vec(Pr))
        end

        local Pr_t,Pr_t! = build(θ,t,φ,Pr) do θ,t,φ,Pr
            riccati(fx(θ,t,φ), Kr(θ,t,φ,Pr), @vec(Pr), Qr(θ,t,φ), Rr(θ,t,φ))
        end

        # add definitions to PRONTO
        PRONTO.Kr(M::$T,θ,t,φ,Pr) = Kr(θ,t,φ,Pr) # NU,NX
        PRONTO.Pr_t(M::$T,θ,t,φ,Pr) = Pr_t(θ,t,φ,Pr) # NX,NX
        PRONTO.Pr_t!(M::$T,buf,θ,t,φ,Pr) = Pr_t!(buf,θ,t,φ,Pr) # NX,NX    

        println("\t > projection solver")

        local ξ_t,ξ_t! = build(θ,t,ξ,φ,Pr) do θ,t,ξ,φ,Pr
            local x,u = split($T(),ξ)
            local α,μ = split($T(),φ)

            vcat(f(θ,t,ξ)...,
                (@vec(μ) - Kr(θ,t,φ,Pr)*(@vec(x) - @vec(α)) - @vec(u))...)
        end
        # add definitions to PRONTO
        PRONTO.ξ_t(M::$T,θ,t,ξ,φ,Pr) = ξ_t(θ,t,ξ,φ,Pr) # NX+NU
        PRONTO.ξ_t!(M::$T,buf,θ,t,ξ,φ,Pr) = ξ_t!(buf,θ,t,ξ,φ,Pr) # NX+NU
        
        println("\t > optimizer solver")

        #Ko = R\(S'+B'P)
        local Ko,Ko! = build(θ,t,ξ,Po) do θ,t,ξ,Po
            luu(θ,t,ξ)\(lxu(θ,t,ξ)' .+ fu(θ,t,ξ)'*@vec(Po))
        end
        PRONTO.Ko(M::$T,θ,t,ξ,Po) = Ko(θ,t,ξ,Po) # NU,NX

        local Po_t,Po_t! = build(θ,t,ξ,Po) do θ,t,ξ,Po
            riccati(fx(θ,t,ξ), Ko(θ,t,ξ,Po), @vec(Po), lxx(θ,t,ξ), luu(θ,t,ξ))
        end
        PRONTO.Po_t(M::$T,θ,t,ξ,Po) = Po_t(θ,t,ξ,Po) # NX,NX
        PRONTO.Po_t!(M::$T,buf,θ,t,ξ,Po) = Po_t!(buf,θ,t,ξ,Po) # NX,NX    

        println("\t > optimizer costate solver")

        # size NX
        local vo,vo! = build(θ,t,ξ,ro) do θ,t,ξ,ro
            -luu(θ,t,ξ)\(fu(θ,t,ξ)'*@vec(ro) + lu(θ,t,ξ))
        end
        PRONTO.vo(M::$T,θ,t,ξ,ro) = vo(θ,t,ξ,ro) # NU,NX

        local ro_t,ro_t! = build(θ,t,ξ,Po,ro) do θ,t,ξ,Po,ro
            costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Ko(θ,t,ξ,Po), @vec(ro))
        end
        PRONTO.ro_t(M::$T,θ,t,ξ,Po,ro) = ro_t(θ,t,ξ,Po,ro) # NX,NX
        PRONTO.ro_t!(M::$T,buf,θ,t,ξ,Po,ro) = ro_t!(buf,θ,t,ξ,Po,ro) # NX,NX 

        println("\t > lagrangian/costate solver")

        # size NX
        local λ_t,λ_t! = build(θ,t,ξ,φ,Pr,λ) do θ,t,ξ,φ,Pr,λ
            costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Kr(θ,t,φ,Pr), @vec(λ))
        end
        PRONTO.λ_t(M::$T,θ,t,ξ,φ,Pr,λ) = λ_t(θ,t,ξ,φ,Pr,λ)
        PRONTO.λ_t!(M::$T,buf,θ,t,ξ,φ,Pr,λ) = λ_t!(buf,θ,t,ξ,φ,Pr,λ)


        println("\t > search direction solver")

        _t_derivation = round((time_ns() - _t0_derivation)/1e9; digits=4)
        println()
        info("model derivation completed in $_t_derivation seconds")
    end
end


# ----------------------------------- ode solution handling ----------------------------------- #
include("utils.jl")

# ----------------------------------- main loop ----------------------------------- #

Pr_ode(dPr, Pr, (M,φ,θ), t) = Pr_t!(M, dPr, θ, t, φ(t), Pr)
ξ_ode(dξ, ξ, (M,θ,φ,Pr), t) = ξ_t!(M, dξ, θ, t, ξ, φ(t), Pr(t))
Po_ode(dPo, Po, (M,ξ,θ), t) = Po_t!(M, dPo, θ, t, ξ(t), Po)


function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ) where {NX,NU,NΘ}
    
    info("solving regulator")
    Pr_f = diagm(ones(NX))
    Pr = Solution(ODEProblem(Pr_ode,Pr_f,(tf,t0),(M,φ,θ)), NX, NX)
    # Pr = Solution(M, Pr_ode, Prf, (t0,tf), (M,φ,θ))

    info("solving projection")
    ξ = Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))
    
    info("solving optimizer")
    Po_f = pxx(M,θ,tf,φ(tf))
    Po = Solution(ODEProblem(Po_ode,Po_f,(tf,t0),(M,ξ,θ)), NX, NX)

    # ro_f
    # ro = Solution()
    return (Pr,ξ,Po)
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