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
    # anonymous fxn
    f_sym = collect(Base.invokelatest(f, args...))
    # symbolic
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
    # symbolic
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

include("model.jl")

# ----------------------------------- ode solution handling ----------------------------------- #
include("odes.jl")

# ----------------------------------- main loop ----------------------------------- #

Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_dt!(M,dPr,θ,t,φ(t),Pr)
ξ_ode(dξ,ξ,(M,θ,φ,Pr),t) = dξ_dt!(M,dξ,θ,t,ξ,φ(t),Pr(t))
λ_ode(dλ,λ,(M,θ,ξ,φ,Pr),t) = dλ_dt!(M,dλ,θ,t,ξ(t),φ(t),Pr(t),λ)
Po_1_ode(dPo,Po,(M,θ,ξ),t) = dPo_dt_1!(M,dPo,θ,t,ξ(t),Po)
Po_2_ode(dPo,Po,(M,θ,ξ,λ),t) = dPo_dt_2!(M,dPo,θ,t,ξ(t),λ(t),Po)
ro_1_ode(dro,ro,(M,θ,ξ,Po),t) = dro_dt_1!(M,dro,θ,t,ξ(t),Po(t),ro)
ro_2_ode(dro,ro,(M,θ,ξ,λ,Po),t) = dro_dt_2!(M,dro,θ,t,ξ(t),λ(t),Po(t),ro)
ζ_1_ode(dζ,ζ,(M,θ,ξ,Po,ro),t) = dζ_dt_1!(M,dζ,θ,t,ξ(t),ζ,Po(t),ro(t))
ζ_2_ode(dζ,ζ,(M,θ,ξ,λ,Po,ro),t) = dζ_dt_2!(M,dζ,θ,t,ξ(t),ζ,λ(t),Po(t),ro(t))
#MAYBE: split y into ω and Ω
y_ode(dy,y,(M,θ,ξ,ζ,λ),t) = dy_dt!(M,dy,θ,t,ξ(t),ζ(t),λ(t))
h_ode(dh,h,(M,θ,ξ),t) = dh_dt!(M,dh,θ,t,ξ(t))
φ̂_ode(dφ̂,φ̂,(M,θ,ξ,φ,ζ,γ,Pr),t) = dφ̂_dt!(M,dφ̂,θ,t,ξ(t),φ(t),ζ(t),φ̂,γ,Pr(t))

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
        Po = ODE(Po_2_ode, Po_f, (tf,t0), (M,θ,ξ,λ), ODEBuffer{Tuple{NX,NX}}(); verbose=false, callback=cb)
        if Po.sln.retcode == :Success
            iinfo("success\n")
            order = 2
            ro = ODE(ro_2_ode, ro_f, (tf,t0), (M,θ,ξ,λ,Po), ODEBuffer{Tuple{NX}}())
            ζ = ODE(ζ_2_ode, ζ0, (t0,tf), (M,θ,ξ,λ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
        else
            iinfo("unstable, switching to 1st order\n")
            order = 1
            Po = ODE(Po_1_ode, Po_f, (tf,t0), (M,θ,ξ), ODEBuffer{Tuple{NX,NX}}())
            ro = ODE(ro_1_ode, ro_f, (tf,t0), (M,θ,ξ,Po), ODEBuffer{Tuple{NX}}())
            ζ = ODE(ζ_1_ode, ζ0, (t0,tf), (M,θ,ξ,Po,ro), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
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


# temporary debugging helpers

function _Qo_2(M)
    # create symbolic variables
    @variables θ[1:nθ(M)]
    @variables t
    @variables x[1:nx(M)] 
    @variables u[1:nu(M)]
    ξ = vcat(x,u)
    @variables α[1:nx(M)] 
    @variables μ[1:nu(M)]
    φ = vcat(α,μ)
    @variables z[1:nx(M)] 
    @variables v[1:nu(M)]
    ζ = vcat(z,v)
    @variables α̂[1:nx(M)] 
    @variables μ̂[1:nu(M)]
    φ̂ = vcat(α̂,μ̂)
    @variables Pr[1:nx(M),1:nx(M)]
    @variables Po[1:nx(M),1:nx(M)]
    @variables ro[1:nx(M)]
    @variables λ[1:nx(M)]
    @variables γ
    @variables y[1:2] #YO: can we separate these into scalar Dh/D2g?
    @variables h #MAYBE: rename j or J?

    lxx(M,θ,t,ξ) + sum(λ[k].*fxx(M,θ,t,ξ)[k,:,:] for k in 1:nx(M))
end



end # module