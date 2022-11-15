module PRONTO

using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
using LinearAlgebra
using UnicodePlots
using MacroTools
using SparseArrays

using DifferentialEquations
using Symbolics
using Symbolics: derivative

export @derive
export pronto
export info

export @tick,@tock,@clock

export ODE, ODEBuffer
export dae
export preview





# ----------------------------------- 0. preliminaries & helpers ----------------------------------- #
include("helpers.jl")
#TODO: define buffer type


# ----------------------------------- 1. model definition ----------------------------------- #
using MacroTools
using MacroTools: @capture

export @model
# export Model
export nx,nu,nθ


abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ

# auto-defines some useful methods for core model functions
# a good place to look for a complete list of PRONTO's generated internal functions



# ----------------------------------- 2. model derivation ----------------------------------- #


function build(f, args...)
    f_sym = collect(Base.invokelatest(f, args...)) # anonymous -> symbolic[]
    f_ex = build_function(f_sym, args...) # symbolic -> expression
    return eval.(f_ex) # expression -> anonymous
end

function jacobian(dx, f, args...; force_dims=nothing)

    f_sym = collect(Base.invokelatest(f, args...)) # anonymous -> symbolic[]
    
    # generate symbolic derivatives
    jac_sym = map(1:length(dx)) do i

        map(f_sym) do f

            derivative(f, dx[i])
        end
    end

    # concatenate nx-long vector of size (dims...) arrys to one size (dims..., nx) array
    fx_sym = cat(jac_sym...; dims=ndims(f_sym)+1)

    isnothing(force_dims) || (fx_sym = reshape(fx_sym, force_dims...))

    fx_ex = build_function(fx_sym, args...) # symbolic -> expression
    return eval.(fx_ex) # expression -> anonymous
end

struct Jacobian
    dx
end
(J::Jacobian)(f, args...; kw...) = jacobian(J.dx, f, args...; kw...)


# ----------------------------------- 3. intermediate operators ----------------------------------- #

function Ar(M::Model{NX,NU,NΘ},θ,t,φ) where {NX,NU,NΘ}
    # make buf = Buffer{NX,NX}
    # fx!(M,buf,θ,t,φ)
    # return buf
end



function _symbolics(T)::Expr
    return quote

        # create symbolic variables
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
        @variables α̂[1:nx($T())] 
        @variables μ̂[1:nu($T())]
        φ̂ = vcat(α̂,μ̂)
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables Po[1:nx($T()),1:nx($T())]
        @variables ro[1:nx($T())]
        @variables λ[1:nx($T())]
        @variables γ
        @variables y[1:2] #YO: can we separate these into scalar Dh/D2g?
        @variables h #MAYBE: rename j or J?

        # create Jacobian operators
        Jx,Ju = Jacobian.([x,u])
    end
end

function _dynamics(T)::Expr

    user_f = esc(:f)

    return quote
        let
        # load user function, remap ξ<->(x,u)
        local f,f! = build(θ,t,ξ) do θ,t,ξ

            local x,u = split($T(),ξ)
            # ($(esc(:(f))))(θ,t,x,u)
            ($user_f)(θ,t,x,u)
        end

        # derive models
        local fx,fx! = Jx(f,θ,t,ξ)
        local fu,fu! = Ju(f,θ,t,ξ)
        local fxx,fxx! = Jx(fx,θ,t,ξ; force_dims=(nx($T()),nx($T()),nx($T())))
        local fxu,fxu! = Ju(fx,θ,t,ξ)
        local fuu,fuu! = Ju(fu,θ,t,ξ)

        # add definitions to PRONTO
        PRONTO.f(M::$T,θ,t,ξ) = f(θ,t,ξ) # NX
        PRONTO.fx(M::$T,θ,t,ξ) = fx(θ,t,ξ) # NX,NX
        PRONTO.fu(M::$T,θ,t,ξ) = fu(θ,t,ξ) # NX,NU
        PRONTO.fxx(M::$T,θ,t,ξ) = fxx(θ,t,ξ) # NX,NX,NX
        PRONTO.fxu(M::$T,θ,t,ξ) = fxu(θ,t,ξ) # NX,NX,NU
        PRONTO.fuu(M::$T,θ,t,ξ) = fuu(θ,t,ξ) # NX,NU,NU
        PRONTO.f!(M::$T,buf,θ,t,ξ) = f!(buf,θ,t,ξ) # NX
    end
    end
end

function _stage_cost(T)::Expr

    return quote
        let
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
    end
    end
end

function _terminal_cost(T)::Expr

    return quote
        let
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
    end
    end
end


# ----------------------------------- 4. ode equations ----------------------------------- #

# ----------------------------------- 5. ode solutions ----------------------------------- #

# ----------------------------------- 6. trajectories ----------------------------------- #



# ----------------------------------- * buffer type ----------------------------------- #
# ----------------------------------- * timing ----------------------------------- #




# ----------------------------------- PRONTO loop ----------------------------------- #



end