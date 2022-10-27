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

export @tick,@tock,@clock

export ODE, ODEBuffer
export dae
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
iinfo(str) = print("    > ", str) # secondary-level

# ----------------------------------- code timing ----------------------------------- #

tick(name) = esc(Symbol(String(name)*"_tick"))
tock(name) = esc(Symbol(String(name)*"_tock"))

macro tick(name=:(_))
    :($(tick(name)) = time_ns())
end

macro tock(name=:(_))
    :($(tock(name)) = time_ns())
end

macro clock(name=:(_))
    _tick = tick(name)
    _tock = tock(name)
    ms = :(($_tock - $_tick)/1e6)
    :("$($:(round($ms; digits=3))) ms")
end

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
    eex = esc(ex)
    :(collect($eex))
end

split(M::Model, ξ) = (ξ[1:nx(M)], ξ[(nx(M)+1):end])

# loads definitions for model M into pronto from autodiff based on current definitions in Main
#FUTURE: break into sections:
    # 1. global macro is spliced from expressions, functions generate these expressions from T
    # 2. global macro simply expands to sub-macros, sub-macros evaluate from T
    # 3. ¿Por qué no los dos?

# _symbolics(::T) where {T<:Model} = _symbolics(T)

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
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables Po[1:nx($T()),1:nx($T())]
        @variables ro[1:nx($T())]
        @variables λ[1:nx($T())]

        # create Jacobian operators
        Jx,Ju = Jacobian.([x,u])
    end
end

function _dynamics(T)::Expr
    user_f = esc(:f)
    return quote
        # load user function, remap ξ<->(x,u)
        local f,f! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            # ($(esc(:(f))))(θ,t,x,u)
            ($user_f)(θ,t,x,u)
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
    end
end

function _stage_cost(T)::Expr
    return quote
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

function _terminal_cost(T)::Expr
    return quote
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

function _regulator(T)::Expr
    return quote
        # load user functions remapping (x,u)->ξ
        local Qr,Qr! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            # ($user_Qr)(θ,t,x,u)
            ($(esc(:(Qr))))(θ,t,x,u)

        end
        local Rr,Rr! = build(θ,t,ξ) do θ,t,ξ
            local x,u = split($T(),ξ)
            ($(esc(:(Rr))))(θ,t,x,u)
        end

        #Kr = Rr\(Br'Pr)
        local Kr,Kr! = build(θ,t,φ,Pr) do θ,t,φ,Pr
            inv(Rr(θ,t,φ))*(fu(θ,t,φ)'*@vec(Pr))
        end

        local Pr_t,Pr_t! = build(θ,t,φ,Pr) do θ,t,φ,Pr
            riccati(fx(θ,t,φ), Kr(θ,t,φ,Pr), @vec(Pr), Qr(θ,t,φ), Rr(θ,t,φ))
        end

        # add definitions to PRONTO
        PRONTO.Rr(M::$T,θ,t,φ) = Rr(θ,t,φ)
        PRONTO.Qr(M::$T,θ,t,φ) = Qr(θ,t,φ)
        PRONTO.Kr(M::$T,θ,t,φ,Pr) = Kr(θ,t,φ,Pr) # NU,NX
        PRONTO.Pr_t(M::$T,θ,t,φ,Pr) = Pr_t(θ,t,φ,Pr) # NX,NX
        PRONTO.Pr_t!(M::$T,buf,θ,t,φ,Pr) = Pr_t!(buf,θ,t,φ,Pr) # NX,NX    
    end    
end

function _projection(T)::Expr
    return quote
        # build dae to solve for dξ/dt
        local ξ_t,ξ_t! = build(θ,t,ξ,φ,Pr) do θ,t,ξ,φ,Pr
            local x,u = split($T(),ξ)
            local α,μ = split($T(),φ)

            vcat(f(θ,t,ξ)...,
                (@vec(μ) - Kr(θ,t,φ,Pr)*(@vec(x) - @vec(α)) - @vec(u))...)
        end

        # add definitions to PRONTO
        PRONTO.ξ_t(M::$T,θ,t,ξ,φ,Pr) = ξ_t(θ,t,ξ,φ,Pr) # NX+NU
        PRONTO.ξ_t!(M::$T,buf,θ,t,ξ,φ,Pr) = ξ_t!(buf,θ,t,ξ,φ,Pr) # NX+NU
    end    
end

function _optimizer(T)::Expr
    B = :(fu(θ,t,ξ))
    return quote

        # ideally:
        # @build Ko (θ,t,ξ,Po) -> inv(Ro)\(S'+B'Po)

        #Ko = Ro\(S'+B'Po)
        local Ko,Ko! = build(θ,t,ξ,Po) do θ,t,ξ,Po
            inv(luu(θ,t,ξ))*(lxu(θ,t,ξ)' .+ ($B)'*@vec(Po))
        end
        PRONTO.Ko(M::$T,θ,t,ξ,Po) = Ko(θ,t,ξ,Po) # NU,NX

        #dPo_dt = -A'Po - Po*A + Ko'Ro*Ko - Qo
        local Po_t,Po_t! = build(θ,t,ξ,Po) do θ,t,ξ,Po
            riccati(fx(θ,t,ξ), Ko(θ,t,ξ,Po), @vec(Po), lxx(θ,t,ξ), luu(θ,t,ξ))
        end

        # add definitions to PRONTO
        PRONTO.Po_t(M::$T,θ,t,ξ,Po) = Po_t(θ,t,ξ,Po) # NX,NX
        PRONTO.Po_t!(M::$T,buf,θ,t,ξ,Po) = Po_t!(buf,θ,t,ξ,Po) # NX,NX
        
        # costate
        local vo,vo! = build(θ,t,ξ,ro) do θ,t,ξ,ro
            inv(-luu(θ,t,ξ))*(fu(θ,t,ξ)'*@vec(ro) + lu(θ,t,ξ))
        end
        
        local ro_t,ro_t! = build(θ,t,ξ,Po,ro) do θ,t,ξ,Po,ro
            costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Ko(θ,t,ξ,Po), @vec(ro))
        end
        
        # add definitions to PRONTO
        PRONTO.vo(M::$T,θ,t,ξ,ro) = vo(θ,t,ξ,ro) # NU,NX
        PRONTO.ro_t(M::$T,θ,t,ξ,Po,ro) = ro_t(θ,t,ξ,Po,ro) # NX
        PRONTO.ro_t!(M::$T,buf,θ,t,ξ,Po,ro) = ro_t!(buf,θ,t,ξ,Po,ro) # NX
    end    
end

function _lagrangian(T)::Expr
    return quote
        local λ_t,λ_t! = build(θ,t,ξ,φ,Pr,λ) do θ,t,ξ,φ,Pr,λ
            costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Kr(θ,t,φ,Pr), @vec(λ))
        end

        # add definitions to PRONTO
        PRONTO.λ_t(M::$T,θ,t,ξ,φ,Pr,λ) = λ_t(θ,t,ξ,φ,Pr,λ) #NX
        PRONTO.λ_t!(M::$T,buf,θ,t,ξ,φ,Pr,λ) = λ_t!(buf,θ,t,ξ,φ,Pr,λ) #NX
    end
end

function _search_direction(T)::Expr
    Ko = :(Ko(θ,t,ξ,Po))
    vo = :(vo(θ,t,ξ,ro))
    A = :(fx(θ,t,ξ))
    B = :(fu(θ,t,ξ))
    v = :(collect(v))
    z = :(collect(z))
    return quote

        local ζ_t,ζ_t! = build(θ,t,ξ,ζ,Po,ro) do θ,t,ξ,ζ,Po,ro
            local z,v = split($T(),ζ)
            vcat(($A*$z + $B*$v)...,
                ($vo - $Ko*$z - $v)...)
        end
        local _v,_v! = build(θ,t,ξ,ζ,Po,ro) do θ,t,ξ,ζ,Po,ro
            local z,v = split($T(),ζ)
            $vo - $Ko*$z
        end
        # add definitions to PRONTO
        PRONTO._v(M::$T,θ,t,ξ,ζ,Po,ro) = _v(θ,t,ξ,ζ,Po,ro)
        PRONTO.ζ_t(M::$T,θ,t,ξ,ζ,Po,ro) = ζ_t(θ,t,ξ,ζ,Po,ro)
        PRONTO.ζ_t!(M::$T,buf,θ,t,ξ,ζ,Po,ro) = ζ_t!(buf,θ,t,ξ,ζ,Po,ro)
    end
end

macro derive(T)
    # make sure we use the local context
    T = esc(T)
    # M = T()
    user_Qr = esc(:Qr)

    return quote
        println()
        info("deriving the $(as_bold("$($T)")) model:")
        @tick derive_time

        # generate symbolic variables for derivation
        iinfo("preparing symbolics ... "); @tick
        $(_symbolics(T)); @tock; println(@clock)

        iinfo("dynamics derivatives ... "); @tick
        $(_dynamics(T)); @tock; println(@clock)

        iinfo("stage cost derivatives ... "); @tick
        $(_stage_cost(T)); @tock; println(@clock)
        
        iinfo("terminal cost derivatives ... "); @tick
        $(_terminal_cost(T)); @tock; println(@clock)

        iinfo("regulator solver ... "); @tick
        $(_regulator(T)); @tock; println(@clock)

        iinfo("projection solver ... "); @tick
        $(_projection(T)); @tock; println(@clock)
        
        iinfo("optimizer solver ... "); @tick
        $(_optimizer(T)); @tock; println(@clock)
        
        iinfo("lagrangian/costate solver ... "); @tick
        $(_lagrangian(T)); @tock; println(@clock)

        iinfo("search direction solver ... "); @tick
        $(_search_direction(T)); @tock; println(@clock)












        @tock derive_time
        info("model derivation completed in $(@clock derive_time)\n")
    end
end
#MAYBE: precompile model equations?

# ----------------------------------- ode solution handling ----------------------------------- #
include("odes.jl")

# ----------------------------------- main loop ----------------------------------- #

Pr_ode(dPr,Pr,(M,θ,φ),t) = Pr_t!(M,dPr,θ,t,φ(t),Pr)
ξ_ode(dξ,ξ,(M,θ,φ,Pr),t) = ξ_t!(M,dξ,θ,t,ξ,φ(t),Pr(t))
Po_ode(dPo,Po,(M,θ,ξ),t) = Po_t!(M,dPo,θ,t,ξ(t),Po)
ro_ode(dro,ro,(M,θ,ξ,Po),t) = ro_t!(M,dro,θ,t,ξ(t),Po(t),ro)
λ_ode(dλ,λ,(M,θ,ξ,φ,Pr),t) = λ_t!(M,dλ,θ,t,ξ(t),φ(t),Pr(t),λ)
ζ_ode(dζ,ζ,(M,θ,ξ,Po,ro),t) = ζ_t!(M,dζ,θ,t,ξ(t),ζ,Po(t),ro(t))
# y_ode()

function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ) where {NX,NU,NΘ}
    
    info(as_bold(string(nameof(typeof(M))))*" model iteration 1:")


    iinfo("regulator ... "); @tick

    Pr_f = diagm(ones(NX))
    Pr = ODE(Pr_ode, Pr_f, (tf,t0), (M,θ,φ), ODEBuffer{Tuple{NX,NX}}())
    
    @tock; println(@clock)
    # println(preview(Pr))


    iinfo("projection ... "); @tick
    # ξ = Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))
    ξ = ODE(ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
    @tock; println(@clock)
    # println(preview(ξ))


    iinfo("optimizer ... "); @tick
    Po_f = pxx(M,θ,tf,φ(tf))
    Po = ODE(Po_ode, Po_f, (tf,t0), (M,θ,ξ), ODEBuffer{Tuple{NX,NX}}())

    ro_f = px(M,θ,tf,φ(tf))
    ro = ODE(ro_ode, ro_f, (tf,t0), (M,θ,ξ,Po), ODEBuffer{Tuple{NX}}())
    @tock; println(@clock)
    # println(preview(Po))
    # println(preview(ro))


    iinfo("lagrangian ... "); @tick
    λ_f = px(M,θ,tf,φ(tf))
    λ = ODE(λ_ode, λ_f, (tf,t0), (M,θ,ξ,φ,Pr), ODEBuffer{Tuple{NX}}())
    @tock; println(@clock)
    # println(preview(λ))
    

    iinfo("search direction ... "); @tick
    ζ0 = [zeros(NX); 0]
    ζ = ODE(ζ_ode, ζ0, (t0,tf), (M,θ,ξ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
    @tock; println(@clock)

    return ζ
end

#MAYBE:
# info(M, "message") -> [PRONTO-TwoSpin: message

include("kernel.jl")
# helpers for finding guess trajectories
include("guess.jl")

end # module