# PRONTO.jl v0.3.0-dev
module PRONTO
# include("kernels.jl")
using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
using LinearAlgebra
using UnicodePlots
using MacroTools
using SparseArrays
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

# inv!(A) = LinearAlgebra.inv!(lu!(A)) # general
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
    isnothing(force_dims) || (fx_sym = reshape(fx_sym, force_dims...))
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
        @variables α̂[1:nx($T())] 
        @variables μ̂[1:nu($T())]
        φ̂ = vcat(α̂,μ̂)
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables Po[1:nx($T()),1:nx($T())]
        @variables ro[1:nx($T())]
        @variables λ[1:nx($T())]
        @variables γ
        @variables y[1:2]
        @variables h #MAYBE: rename j or J?

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
        PRONTO.Kr(M::$T,θ,t,φ,Pr) = Kr(θ,t,φ,Pr) # NU,NX
        PRONTO.Qr(M::$T,θ,t,φ) = Qr(θ,t,φ)
        PRONTO.Rr(M::$T,θ,t,φ) = Rr(θ,t,φ)
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
            return vcat(

                f(θ,t,ξ)...,
                (@vec(μ) - Kr(θ,t,φ,Pr)*(@vec(x) - @vec(α)) - @vec(u))...
            )
        end

        # add definitions to PRONTO
        PRONTO.ξ_t(M::$T,θ,t,ξ,φ,Pr) = ξ_t(θ,t,ξ,φ,Pr) # NX+NU
        PRONTO.ξ_t!(M::$T,buf,θ,t,ξ,φ,Pr) = ξ_t!(buf,θ,t,ξ,φ,Pr) # NX+NU
    end    
end

# @define $T Ko (θ,t,ξ,Po)->inv($Ro)*(($So)'+($B)'($Po))
# function define(T, ex)
#     @capture(ex, name_(args__)=body_)
#     name! = Symbol(String(name)*"!")
#     return quote
#         local $name,$name! = build($(args...)) do $(args...)
#             $body
#         end
#         PRONTO.$name(M::$T,$(args...)) = ($name)($(args...))
#         PRONTO.$name!(M::$T,buf,$(args...)) = ($name!)(buf,$(args...))
#     end
# end


function _lagrangian(T)::Expr
    A = :(fx(θ,t,ξ))
    B = :(fu(θ,t,ξ))
    a = :(lx(θ,t,ξ))
    b = :(lu(θ,t,ξ))
    Kr = :(Kr(θ,t,φ,Pr))
    λ = :(collect(λ))

    return quote

        local λ_t,λ_t! = build(θ,t,ξ,φ,Pr,λ) do θ,t,ξ,φ,Pr,λ

            -($A-$B*$Kr)'*$λ - $a + ($Kr)'*$b
            # costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Kr(θ,t,φ,Pr), @vec(λ))
        end
        # add definitions to PRONTO
        PRONTO.λ_t(M::$T,θ,t,ξ,φ,Pr,λ) = λ_t(θ,t,ξ,φ,Pr,λ) #NX
        PRONTO.λ_t!(M::$T,buf,θ,t,ξ,φ,Pr,λ) = λ_t!(buf,θ,t,ξ,φ,Pr,λ) #NX
    end
end


function _optimizer(T)::Expr
    A = :(fx(θ,t,ξ))
    B = :(fu(θ,t,ξ))
    Qo1 = :(lxx(θ,t,ξ))
    Ro1 = :(luu(θ,t,ξ))
    So1 = :(lxu(θ,t,ξ))
    Qo2 = :(lxx(θ,t,ξ) + sum(λ[k]*fxx(θ,t,ξ)[k,:,:] for k in 1:nx($T())))
    Ro2 = :(luu(θ,t,ξ) + sum(λ[k]*fuu(θ,t,ξ)[k,:,:] for k in 1:nx($T())))
    So2 = :(lxu(θ,t,ξ) + sum(λ[k]*fxu(θ,t,ξ)[k,:,:] for k in 1:nx($T())))
    Po = :(collect(Po))
    # Ko = define(T, :(Ko(θ,t,ξ,Po) = inv($Ro)*(($So)'+($B)'*($Po))))
    # So/Ro/Qo can be newton or gradient
    return quote

        # ideally:
        # @build Ko (θ,t,ξ,Po) -> inv(Ro)(S'+B'Po)
        # @define $T Ko(θ,t,ξ,Po) = inv($Ro)*(($So)'+($B)'($Po))
        # #Ko = Ro\(So'+B'Po)
        local Ko1,Ko1! = build(θ,t,ξ,Po) do θ,t,ξ,Po
            inv($Ro1)*(($So1)'.+($B)'*($Po))
        end

        #dPo_dt = -A'Po - Po*A + Ko'Ro*Ko - Qo
        local Po1_t,Po1_t! = build(θ,t,ξ,Po) do θ,t,ξ,Po
            riccati($A, Ko1(θ,t,ξ,Po), $Po, $Qo1, $Ro1)
        end

        # add definitions to PRONTO
        PRONTO.Ko1(M::$T,θ,t,ξ,Po) = Ko1(θ,t,ξ,Po) # NU,NX
        PRONTO.Po1_t(M::$T,θ,t,ξ,Po) = Po1_t(θ,t,ξ,Po) # NX,NX
        PRONTO.Po1_t!(M::$T,buf,θ,t,ξ,Po) = Po1_t!(buf,θ,t,ξ,Po) # NX,NX

        local Ko2,Ko2! = build(θ,t,ξ,λ,Po) do θ,t,ξ,λ,Po
            inv($Ro2)*(($So2)'.+($B)'*($Po))
        end

        #NOTE: this is where second order may break
        #dPo_dt = -A'Po - Po*A + Ko'Ro*Ko - Qo
        local Po2_t,Po2_t! = build(θ,t,ξ,λ,Po) do θ,t,ξ,λ,Po
            riccati($A, Ko2(θ,t,ξ,λ,Po), $Po, $Qo2, $Ro2)
        end

        # add definitions to PRONTO
        PRONTO.Ko2(M::$T,θ,t,ξ,λ,Po) = Ko2(θ,t,ξ,λ,Po) # NU,NX
        PRONTO.Po2_t(M::$T,θ,t,ξ,λ,Po) = Po2_t(θ,t,ξ,λ,Po) # NX,NX
        PRONTO.Po2_t!(M::$T,buf,θ,t,ξ,λ,Po) = Po2_t!(buf,θ,t,ξ,λ,Po) # NX,NX
        
        # costate
        local vo1,vo1! = build(θ,t,ξ,ro) do θ,t,ξ,ro
            inv(-$Ro1)*(fu(θ,t,ξ)'*@vec(ro) + lu(θ,t,ξ))
        end
        
        local ro1_t,ro1_t! = build(θ,t,ξ,Po,ro) do θ,t,ξ,Po,ro
            costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Ko1(θ,t,ξ,Po), @vec(ro))
        end

        # add definitions to PRONTO
        PRONTO.vo1(M::$T,θ,t,ξ,ro) = vo1(θ,t,ξ,ro) # NU,NX
        PRONTO.ro1_t(M::$T,θ,t,ξ,Po,ro) = ro1_t(θ,t,ξ,Po,ro) # NX
        PRONTO.ro1_t!(M::$T,buf,θ,t,ξ,Po,ro) = ro1_t!(buf,θ,t,ξ,Po,ro) # NX
        

        local vo2,vo2! = build(θ,t,ξ,λ,ro) do θ,t,ξ,λ,ro
            inv(-$Ro2)*(fu(θ,t,ξ)'*@vec(ro) + lu(θ,t,ξ))
        end

        local ro2_t,ro2_t! = build(θ,t,ξ,λ,Po,ro) do θ,t,ξ,λ,Po,ro
            costate(fx(θ,t,ξ), fu(θ,t,ξ), lx(θ,t,ξ), lu(θ,t,ξ), Ko2(θ,t,ξ,λ,Po), @vec(ro))
        end
        
        # add definitions to PRONTO
        PRONTO.vo2(M::$T,θ,t,ξ,λ,ro) = vo2(θ,t,ξ,λ,ro) # NU,NX
        PRONTO.ro2_t(M::$T,θ,t,ξ,λ,Po,ro) = ro2_t(θ,t,ξ,λ,Po,ro) # NX
        PRONTO.ro2_t!(M::$T,buf,θ,t,ξ,λ,Po,ro) = ro2_t!(buf,θ,t,ξ,λ,Po,ro) # NX
    end    
end

function _search_direction(T)::Expr

    Ko1 = :(Ko1(θ,t,ξ,Po))
    Ko2 = :(Ko2(θ,t,ξ,λ,Po))
    vo1 = :(vo1(θ,t,ξ,ro))
    vo2 = :(vo2(θ,t,ξ,λ,ro))
    A = :(fx(θ,t,ξ))
    B = :(fu(θ,t,ξ))
    v = :(collect(v))
    z = :(collect(z))
    
    return quote

        local ζ1_t,ζ1_t! = build(θ,t,ξ,ζ,Po,ro) do θ,t,ξ,ζ,Po,ro

            local z,v = split($T(),ζ)
            vcat(
                
                ($A*$z + $B*$v)...,
                ($vo1 - $Ko1*$z - $v)...
            )
        end
        PRONTO.ζ1_t(M::$T,θ,t,ξ,ζ,Po,ro) = ζ1_t(θ,t,ξ,ζ,Po,ro)
        PRONTO.ζ1_t!(M::$T,buf,θ,t,ξ,ζ,Po,ro) = ζ1_t!(buf,θ,t,ξ,ζ,Po,ro)

        local ζ2_t,ζ2_t! = build(θ,t,ξ,ζ,λ,Po,ro) do θ,t,ξ,ζ,λ,Po,ro

            local z,v = split($T(),ζ)
            vcat(
                
                ($A*$z + $B*$v)...,
                ($vo2 - $Ko2*$z - $v)...
            )
        end
        PRONTO.ζ2_t(M::$T,θ,t,ξ,ζ,λ,Po,ro) = ζ2_t(θ,t,ξ,ζ,λ,Po,ro)
        PRONTO.ζ2_t!(M::$T,buf,θ,t,ξ,ζ,λ,Po,ro) = ζ2_t!(buf,θ,t,ξ,ζ,λ,Po,ro)


        local _v,_v! = build(θ,t,ξ,ζ,Po,ro) do θ,t,ξ,ζ,Po,ro
            
            local z,v = split($T(),ζ)
            $vo1 - $Ko1*$z
        end
        PRONTO._v(M::$T,θ,t,ξ,ζ,Po,ro) = _v(θ,t,ξ,ζ,Po,ro)
    end
end



function _cost_derivatives(T)::Expr
    a = :(lx(θ,t,ξ))
    b = :(lu(θ,t,ξ))
    v = :(collect(v))
    z = :(collect(z))
    # always second order
    Qo = :(lxx(θ,t,ξ) + sum(λ[k]*fxx(θ,t,ξ)[k,:,:] for k in 1:nx($T())))
    Ro = :(luu(θ,t,ξ) + sum(λ[k]*fuu(θ,t,ξ)[k,:,:] for k in 1:nx($T())))
    So = :(lxu(θ,t,ξ) + sum(λ[k]*fxu(θ,t,ξ)[k,:,:] for k in 1:nx($T())))

    return quote
 

        # simply need dy/dt
        # @build y_t (θ,t,ξ,ζ) begin
        local y_t, y_t! = build(θ,t,ξ,ζ,λ) do θ,t,ξ,ζ,λ
        
            local z, v = split($T(),ζ)
            vcat(

                ($a)'*($z) + ($b)'*($v),
                ($z)'*($Qo)*($z) + 2*($z)'*($So)*($v) + ($v)'*($Ro)*($v)
            )
        end
        PRONTO.y_t(M::$T,θ,t,ξ,ζ,λ) = y_t(θ,t,ξ,ζ,λ)
        PRONTO.y_t!(M::$T,buf,θ,t,ξ,ζ,λ) = y_t!(buf,θ,t,ξ,ζ,λ)


        local Dh, Dh! = build(θ,t,φ,ζ,y) do θ,t,φ,ζ,y

            local z, v = split($T(),ζ)
            y[1] + (px(θ,t,φ))'*($z)
        end
        PRONTO._Dh(M::$T,θ,t,φ,ζ,y) = Dh(θ,t,φ,ζ,y)

        local D2g, D2g! = build(θ,t,φ,ζ,y) do θ,t,φ,ζ,y

            local z, v = split($T(),ζ)
            y[2] + ($z)'*pxx(θ,t,φ)*($z)
        end
        PRONTO._D2g(M::$T,θ,t,φ,ζ,y) = D2g(θ,t,φ,ζ,y)
    end
end

function _armijo(T)::Expr
    Kr = :(Kr(θ,t,φ,Pr))
    u = :(collect(u)) 
    v = :(collect(v)) 
    μ̂ = :(collect(μ̂))   
    x = :(collect(x)) 
    z = :(collect(z)) 
    α̂ = :(collect(α̂))
    return quote
        
        # φ̂ = ξ+γζ

        # # projection with respect to γ
        local φ̂_t, φ̂_t! = build(θ,t,ξ,φ,ζ,φ̂,γ,Pr) do θ,t,ξ,φ,ζ,φ̂,γ,Pr
            
            local α, μ = split($T(),φ)
            local x, u = split($T(),ξ)
            local z, v = split($T(),ζ)
            local α̂, μ̂ = split($T(),φ̂)

            return vcat(
                f(θ,t,φ̂)...,
                ($u + γ*$v) - ($Kr)*($α̂ - ($x + γ*$z)) - $μ̂...
            )
        end
        PRONTO.φ̂_t(M::$T,θ,t,ξ,φ,ζ,φ̂,γ,Pr) = φ̂_t(θ,t,ξ,φ,ζ,φ̂,γ,Pr)
        PRONTO.φ̂_t!(M::$T,buf,θ,t,ξ,φ,ζ,φ̂,γ,Pr) = φ̂_t!(buf,θ,t,ξ,φ,ζ,φ̂,γ,Pr)
 

        # cost function ode
        local h_t, h_t! = build(θ,t,ξ) do θ,t,ξ

            l(θ,t,ξ)
        end
        PRONTO.h_t(M::$T,θ,t,ξ) = h_t(θ,t,ξ)
        PRONTO.h_t!(M::$T,buf,θ,t,ξ) = h_t!(buf,θ,t,ξ)
    end
end

macro derive(T)

    # make sure we use the local context
    T = esc(T)

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

        iinfo("cost derivative solver ... "); @tick
        $(_cost_derivatives(T)); @tock; println(@clock)
        
        iinfo("armijo rule ... "); @tick
        $(_armijo(T)); @tock; println(@clock)

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
λ_ode(dλ,λ,(M,θ,ξ,φ,Pr),t) = λ_t!(M,dλ,θ,t,ξ(t),φ(t),Pr(t),λ)
Po1_ode(dPo,Po,(M,θ,ξ),t) = Po1_t!(M,dPo,θ,t,ξ(t),Po)
Po2_ode(dPo,Po,(M,θ,ξ,λ),t) = Po2_t!(M,dPo,θ,t,ξ(t),λ(t),Po)
ro1_ode(dro,ro,(M,θ,ξ,Po),t) = ro1_t!(M,dro,θ,t,ξ(t),Po(t),ro)
ro2_ode(dro,ro,(M,θ,ξ,λ,Po),t) = ro2_t!(M,dro,θ,t,ξ(t),λ(t),Po(t),ro)
ζ1_ode(dζ,ζ,(M,θ,ξ,Po,ro),t) = ζ1_t!(M,dζ,θ,t,ξ(t),ζ,Po(t),ro(t))
ζ2_ode(dζ,ζ,(M,θ,ξ,λ,Po,ro),t) = ζ2_t!(M,dζ,θ,t,ξ(t),ζ,λ(t),Po(t),ro(t))
#MAYBE: split y into ω and Ω
y_ode(dy,y,(M,θ,ξ,ζ,λ),t) = y_t!(M,dy,θ,t,ξ(t),ζ(t),λ(t))
h_ode(dh,h,(M,θ,ξ),t) = h_t!(M,dh,θ,t,ξ(t))
φ̂_ode(dφ̂,φ̂,(M,θ,ξ,φ,ζ,γ,Pr),t) = φ̂_t!(M,dφ̂,θ,t,ξ(t),φ(t),ζ(t),φ̂,γ,Pr(t))

# for debug:
wait_for_key() = (print(stdout, "press a key..."); read(stdin, 1); nothing)

struct InstabilityError <: Exception
end

# eigcheck(Po,_,_) = maximum(eigvals( ishermitian(Po) ? Po : collect(Po)) ) >= 1e5
function eigcheck(Po,_,_)
    eig = maximum(eigvals( ishermitian(Po) ? Po : collect(Po)) )
    eig >= 1e10 && println(stdout, "max eig = $eig")
    eig >= 1e12
end

# function eigcheck(_,_,integrator)
#     if
# end
export asymmetry
function asymmetry(A)
    (m,n) = size(A)
    @assert m == n "must be square matrix"
    sum([0.5*abs(A[i,j]-A[j,i]) for i in 1:n, j in 1:n])
end
function asymcheck(Po,_,_)
    # asymmetry(Po) > 1e-6 && println(stdout, "\tasymmetry = $(asymmetry(Po))")
    asymmetry(Po) > 1e-6
end
# unstable!(_) = throw(InstabilityError())
unstable!(_) = nothing


function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ; tol = 1e-5, maxiters = 20) where {NX,NU,NΘ}
    #parameters
    # debug/verbose
    # tol = 1e-5
    # maxiters = 10

    for i in 1:maxiters

        cb = DiscreteCallback(asymcheck,unstable!)

        info(as_bold(string(nameof(typeof(M))))*" model iteration $i:")
        @tick iteration

        iinfo("regulator ... "); @tick
        Pr_f = diagm(ones(NX))
        Pr = ODE(Pr_ode, Pr_f, (tf,t0), (M,θ,φ), ODEBuffer{Tuple{NX,NX}}(), callback=cb)
        @tock; println(@clock)


        iinfo("projection ... "); @tick
        # ξ = Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))
        ξ = ODE(ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
        @tock; println(@clock)
        plot_trajectory(M,ξ)
        
        iinfo("lagrangian ... "); @tick
        λ_f = collect(px(M,θ,tf,φ(tf)))
        λ = ODE(λ_ode, λ_f, (tf,t0), (M,θ,ξ,φ,Pr), ODEBuffer{Tuple{NX}}())
        @tock; println(@clock)


        # iinfo("optimizer ... "); @tick
        Po_f = collect(pxx(M,θ,tf,φ(tf)))
        ro_f = collect(px(M,θ,tf,φ(tf)))
        ζ0 = [zeros(NX); zeros(NU)] # TODO: v(0) = vo(0)

        iinfo("trying 2nd order optimizer ... ")
        Po = ODE(Po2_ode, Po_f, (tf,t0), (M,θ,ξ,λ), ODEBuffer{Tuple{NX,NX}}(); verbose=false, callback=cb)
        if Po.sln.retcode == :Success
            iinfo("success\n")
            order = 2
            ro = ODE(ro2_ode, ro_f, (tf,t0), (M,θ,ξ,λ,Po), ODEBuffer{Tuple{NX}}())
            ζ = ODE(ζ2_ode, ζ0, (t0,tf), (M,θ,ξ,λ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
        else
            iinfo("unstable, switching to 1st order\n")
            order = 1
            Po = ODE(Po1_ode, Po_f, (tf,t0), (M,θ,ξ), ODEBuffer{Tuple{NX,NX}}())
            ro = ODE(ro1_ode, ro_f, (tf,t0), (M,θ,ξ,Po), ODEBuffer{Tuple{NX}}())
            ζ = ODE(ζ1_ode, ζ0, (t0,tf), (M,θ,ξ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
        end


        # @tock; println(@clock)
        
        # iinfo("1st order override\n")
        # order = 1
        # Po = ODE(Po1_ode, Po_f, (tf,t0), (M,θ,ξ), ODEBuffer{Tuple{NX,NX}}())
        # ro = ODE(ro1_ode, ro_f, (tf,t0), (M,θ,ξ,Po), ODEBuffer{Tuple{NX}}())
        # ζ = ODE(ζ1_ode, ζ0, (t0,tf), (M,θ,ξ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))

        # iinfo("search direction ... "); @tick
        # ζ0 = [zeros(NX); zeros(NU)] # TODO: v(0) = vo(0)
        # ζ = ODE(ζ_ode, ζ0, (t0,tf), (M,θ,ξ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
        # @tock; println(@clock)


        iinfo("cost derivatives ... "); @tick
        y0 = [0;0]
        y = ODE(y_ode, y0, (t0,tf), (M,θ,ξ,ζ,λ), ODEBuffer{Tuple{2}}())
        Dh = _Dh(M,θ,tf,φ(tf),ζ(tf),y(tf))[]
        @tock; println(@clock)
        iinfo(as_bold("Dh = $(Dh)\n"))
        Dh > 0 && (info("increased cost - quitting"); (return φ))
        -Dh < tol && (info(as_bold("PRONTO converged")); (return φ))

        # compute cost
        hf = p(M,θ,tf,ξ(tf))[]
        h = ODE(h_ode, [0.0], (t0,tf), (M,θ,ξ), ODEBuffer{Tuple{1}}())(tf)[] + hf
        iinfo(as_bold("h = $(h)\n"))


        iinfo("armijo backstep ... \n"); @tick
        local φ̂
        γ = 1.0; α=0.4; β=0.7
        while γ > β^25

            φ̂ = ODE(φ̂_ode, [x0;u0], (t0,tf), (M,θ,ξ,φ,ζ,γ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
        
            # compute cost
            gf = p(M,θ,tf,φ̂(tf))[]
            g = ODE(h_ode, [0.0], (t0,tf), (M,θ,φ̂), ODEBuffer{Tuple{1}}())(tf)[] + gf
            
            # check armijo rule
            iinfo("γ = $γ   h - g = $(h-g) g = $g\n")
            h-g >= -α*γ*Dh ? break : (γ *= β)
        end
        γ <= β^15 && @warn "armijo maxiters"
        φ = φ̂
        @tock; println(@clock)


        @tock iteration
        info("iteration $i took "*(@clock iteration))

    end
    return φ
    # return φ
end

#MAYBE:
# info(M, "message") -> [PRONTO-TwoSpin: message

include("kernel.jl")
# helpers for finding guess trajectories
include("guess.jl")

end # module