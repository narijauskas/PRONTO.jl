# PRONTO.jl dev_0.4
module PRONTO

using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
export @closure

# lu = PRONTO.lu
# function lu end
import LinearAlgebra
using LinearAlgebra: mul!, I
using UnicodePlots
using MacroTools
using SparseArrays
using MatrixEquations

# using OrdinaryDiffEq
using DifferentialEquations
using Symbolics
using Symbolics: derivative
using SymbolicUtils.Code

using ThreadTools

export @derive
export pronto
export info

export @tick,@tock,@clock

export ODE, Buffer
export preview


using Dates: now

using MacroTools
using MacroTools: @capture

using Interpolations
# using MakieCore
# MakieCore.convert_arguments(P::PointBased, x::MyType) = convert_arguments(P, time vector, vector of sampled vectors)
using Base: OneTo
import Base: extrema, length, eachindex, show, size, eltype

# ----------------------------------- #. preliminaries & typedefs ----------------------------------- #

export Model
export nx,nu,nθ

abstract type Model{NX,NU,NΘ} <: FieldVector{NΘ,Float64} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ

nx(::Type{<:Model{NX,NU,NΘ}}) where {NX,NU,NΘ} = NX
nu(::Type{<:Model{NX,NU,NΘ}}) where {NX,NU,NΘ} = NU
nθ(::Type{<:Model{NX,NU,NΘ}}) where {NX,NU,NΘ} = NΘ

# not used
inv!(A) = LinearAlgebra.inv!(LinearAlgebra.cholesky!(Hermitian(A)))


# ----------------------------------- #. helpers ----------------------------------- #
include("helpers.jl")


#YO: can I actually deprecate this? :)
views(::Model{NX,NU,NΘ},ξ) where {NX,NU,NΘ} = (@view ξ[1:NX]),(@view ξ[NX+1:end])


# ----------------------------------- #. model functions ----------------------------------- #
# missing implementations provided by codegen


# or just throw a method error?
struct ModelDefError <: Exception
    θ::Model
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.θ)
    print(io, "PRONTO is missing method definitions for the $T model.\n")
end



f!(dx,x,u,t,θ) = throw(ModelDefError(θ))

Q(α,μ,t,θ) = throw(ModelDefError(θ))
R(α,μ,t,θ) = throw(ModelDefError(θ))

f(x,u,t,θ) = throw(ModelDefError(θ))
fx(x,u,t,θ) = throw(ModelDefError(θ))
fu(x,u,t,θ) = throw(ModelDefError(θ))
# fxx(x,u,t,θ) = throw(ModelDefError(θ))
# fxu(x,u,t,θ) = throw(ModelDefError(θ))
# fuu(x,u,t,θ) = throw(ModelDefError(θ))

l(x,u,t,θ) = throw(ModelDefError(θ))
lx(x,u,t,θ) = throw(ModelDefError(θ))
lu(x,u,t,θ) = throw(ModelDefError(θ))
lxx(x,u,t,θ) = throw(ModelDefError(θ))
lxu(x,u,t,θ) = throw(ModelDefError(θ))
luu(x,u,t,θ) = throw(ModelDefError(θ))

p(x,u,t,θ) = throw(ModelDefError(θ))
px(x,u,t,θ) = throw(ModelDefError(θ))
pxx(x,u,t,θ) = throw(ModelDefError(θ))

# L(λ,x,u,t,θ) = throw(ModelDefError(θ))
# Lx(λ,x,u,t,θ) = throw(ModelDefError(θ))
# Lu(λ,x,u,t,θ) = throw(ModelDefError(θ))
Lxx(λ,x,u,t,θ) = throw(ModelDefError(θ))
Lxu(λ,x,u,t,θ) = throw(ModelDefError(θ))
Luu(λ,x,u,t,θ) = throw(ModelDefError(θ))




# definitions for these must be generated from a user-specified model
include("codegen.jl") # takes derivatives




# ----------------------------------- #. components ----------------------------------- #




include("odes.jl") # ODE solution handling





# Kr(α,μ,Pr,t,θ) = R(α,μ,t,θ)\(fu(α,μ,t,θ)'Pr)





# ----------------------------------- #. regulator  ----------------------------------- #
include("regulator.jl")











# ----------------------------------- #. projection ----------------------------------- #
include("projection.jl")













# ----------------------------------- #. ode functions ----------------------------------- #


# riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q

# function riccati!(out,A,K,P,Q,R)
#     out .= .- Q
#     mul!(out, A', P, -1, 1) # -A'P
#     mul!(out, P, A, -1, 1) # -P*A
#     # can we more efficiently solve: P'B*(R\B'P) ?
#     out .+= K'*R*K
# end


# # forced
# function dx_dt_ol!(dx,x,(θ,μ),t)
#     u = μ(t)
#     f!(dx,x,u,t,θ)
# end

# # regulated
# function dx_dt!(dx,x,(θ,α,μ,Pr),t)
#     α = α(t)
#     μ = μ(t)
#     u = μ - Kr(α,μ,Pr(t),t,θ)*(x-α)
#     f!(dx,x,u,t,θ)
# end

# function dx_dt2!(dx,x,(θ,α,μ,Kr),t)
#     α = α(t)
#     μ = μ(t)
#     Kr = Kr(α,μ,t)
#     u = μ - Kr*(x-α)
#     f!(dx,x,u,t,θ)
# end

# function dξ_dt!(dξ, ξ, (θ,α,μ,Kr), t)
#     α = α(t)
#     μ = μ(t)
#     x,u = views(θ, ξ)
#     dx,du = views(θ, dξ)
#     f!(dx,x,u,t,θ)
#     du .= μ - Kr(t)*(x-α) - u
# end

# export dξ_dt!


# ----------------------------------- #. search direction ----------------------------------- #


# dλ_dt!(dλ, λ, (θ,ξ,Kr), t) = dλ_dt!(dλ, λ, ξ(t)..., Kr(t), t, θ)
# function dλ_dt!(dλ,λ,x,u,Kr,t,θ)
include("optimizer.jl")

include("armijo.jl")
# u_ol(θ,μ,t) = μ(t)
# u_cl(θ,x,α,μ,Pr,t) = μ - Kr(θ,α,μ,Pr,t)*(x-α)

# ----------------------------------- pronto loop ----------------------------------- #

# export dx_dt!
# # export dx_dt_2o!
# export dx_dt_ol!
# export dPr_dt!
fwd(τ) = extrema(τ)
bkwd(τ) = reverse(fwd(τ))

# solve_forward(fxn, x0, p, τ; kw...)
# solve_backward(fxn, x0, p, τ; kw...)

# solves for x(t),u(t)'
function pronto(θ::Model{NX,NU,NΘ}, x0::StaticVector, φ, τ; γmax=1.0,tol = 1e-5, maxiters = 20,verbose=true) where {NX,NU,NΘ}
    t0,tf = τ

    for i in 1:maxiters
        info(i, "iteration")
        # -------------- build regulator -------------- #
        # α,μ -> Kr,x,u
        verbose && iinfo("regulator")
        Kr = regulator(θ, φ, τ)
        verbose && iinfo("projection")
        ξ = projection(θ, x0, φ, Kr, τ)

        # -------------- search direction -------------- #
        # Kr,x,u -> z,v


        verbose && iinfo("lagrangian")
        λ = lagrangian(θ,ξ,φ,Kr,τ)
        verbose && iinfo("optimizer")
        Ko = optimizer(θ,λ,ξ,φ,τ)
        verbose && iinfo("using $(is2ndorder(Ko) ? "2nd" : "1st") order search")
        verbose && iinfo("costate")
        vo = costate(θ,λ,ξ,φ,Ko,τ)
        verbose && iinfo("search_direction")
        ζ = search_direction(θ,ξ,Ko,vo,τ)

        # -------------- cost/derivatives -------------- #
        verbose && iinfo("cost/derivs")

        Dh,D2g = cost_derivs(θ,λ,φ,ξ,ζ,τ)
        
        Dh > 0 && (info("increased cost - quitting"); (return φ))
        -Dh < tol && (info(as_bold("PRONTO converged")); (return φ))

        # compute cost
        h = cost(ξ, τ)
        # verbose && iinfo(as_bold("h = $(h)\n"))
        info("Dh = $Dh, h = $h")

        # -------------- select γ -------------- #

        γ = γmax; aα=0.4; aβ=0.7
        local η
        while γ > aβ^25
            verbose && iinfo("armijo γ = $(round(γ; digits=6))")
            η = armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)
            g = cost(η, τ)
            h-g >= -aα*γ*Dh ? break : (γ *= aβ)
        end
        φ = η
    end
    return φ
end


# @build $T dφ̂_dt(M,θ,t,ξ,φ,ζ,φ̂,γ,Pr) -> vcat(

#     PRONTO.f($M,θ,t,φ̂)...,
#     ($u + γ*$v) - ($Kr)*($α̂ - ($x + γ*$z)) - $μ̂...
# )
# @build $T dh_dt(M,θ,t,ξ) -> PRONTO.l($M,θ,t,ξ)


function cost_derivs(θ,λ,φ,ξ,ζ,τ)
    t0,tf = τ

    yf = solve(ODEProblem(dy_dt, 0, (t0,tf), (θ,ξ,ζ)), Tsit5(); reltol=1e-7)(tf)
    yyf = solve(ODEProblem(dyy_dt, 0, (t0,tf), (θ,λ,ξ,ζ)), Tsit5(); reltol=1e-7)(tf)

    zf = ζ.x(tf)
    αf = φ.x(tf)
    μf = φ.u(tf)
    rf = px(αf,μf,tf,θ)
    Pf = pxx(αf,μf,tf,θ)
    Dh = yf + rf'zf
    D2g = yyf + zf'Pf*zf
    return Dh,D2g
end

function dy_dt(y, (θ,ξ,ζ), t)
    x = ξ.x(t)
    u = ξ.u(t)
    z = ζ.x(t)
    v = ζ.u(t)
    a = lx(x,u,t,θ)
    b = lu(x,u,t,θ)
    return a'z + b'v
end

function dyy_dt(yy, (θ,λ,ξ,ζ), t)
    x = ξ.x(t)
    u = ξ.u(t)
    z = ζ.x(t)
    v = ζ.u(t)
    λ = λ(t)
    Q = Lxx(λ,x,u,t,θ)
    S = Lxu(λ,x,u,t,θ)
    R = Luu(λ,x,u,t,θ)
    return z'Q*z + 2*z'S*v + v'R*v
end

#cost = integral(l(ξ(t))) + p(ξ(tf))

# Regulator(θ,T,α,μ)


# using LinearAlgebra: I

# alternatively:

#bonus: these, along with ODE, can share trait or timeseries supertype
# struct Regulator{NX,NU,NΘ} <: Timeseries{NU,NX}
#     θ::Model{NX,NU,NΘ}
#     α::Timeseries{NX}
#     μ::Timeseries{NU}
#     Pr::Timeseries{NX,NX}
# end

# export TimeDomain
# struct TimeDomain{T}
#     t0::T
#     # dt::T
#     tf::T
# end

# domain(T::TimeDomain) = (T.t0,T.tf)

#TODO: iterate


# export Regulator

# struct Regulator{T,T1,T2,T3}
#     θ::T
#     α::T1
#     μ::T2
#     Pr::T3
# end

# abstract type Timeseries end
# Base.show(io::IO, x::Timeseries) = println(io, preview(x))
# domain(x::Timeseries) = domain(x.T)


# Ko = Ro\(So' + Bo'Po)



# Ko ~ x,u,Po
# Kr ~ α,μ,Pr


# Kr = Regulator(θ,α,μ,Pr) # captures pointers to α,μ,Pr
# Kr(t) # evaluates Kr(θ, α(t), μ(t), Pr(t)) from those pointers and t
# u = ControlInput(θ,x,Kr) # neatly chains thru Kr

# builds lazy eval chain
# vs calling from top down
# vs lazy chain + cache @ time

#=
    options:

    1. build lazy eval chain
        - well defined dependencies
        - convenient behaviors like Kr(t)
        - may double compute x(t), u(t))
        - more work to implement, may allow for granular optimization
    1b. closures
        - allow for convenient chaining implementation
        - but, need to be recompiled
        - unintuitive
    2. call top down (should be very efficient)
        - requires some very long call chains
        - harder to tell what's going on
        - could be fixed with more functions
    3. lazy eval chain with cache
        - duplicated calls wouls always be at same time
        - should get benefits of 1 and 2
        - most work to implement

=#


# ----------------------------------- ?? ----------------------------------- #


# M contains buffers
# @inline Ar(M,θ,t,φ) = (fx!(M, M.Ar, M.θ, t, φ); return M.Ar)
# @inline Br(M,θ,t,φ) = (fu!(M, M.Br, M.θ, t, φ); return M.Br)
# @inline Qr(M,θ,t,φ) = (Qrr!(M, M.Qr, M.θ, t, φ); return M.Qr)
# @inline Rr(M,θ,t,φ) = (Rrr!(M, M.Rr, M.θ, t, φ); return M.Rr)

# regulator(B,P,R) = Diagonal(R)\B'P

# riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q
# M.buf_nx_nx - could this cause problems, eg. with threading?



# generate naive, generic Kr, and dPr_dt
# function Kr(M, θ, t, φ, Pr)
#     mul!(out, Diagonal(R)\B', Pr)
#     M.Kr .= Diagonal(Rr(M,θ,t,φ)) \ (Br(M,θ,t,φ)'*Pr)
#     return M.Kr
# end


# # dPr_dt!(M::Model, out, θ, t, φ, Pr) = out .= riccati(Ar(M,θ,t,φ), Kr(M,θ,t,φ,Pr), Pr, Qr(M,θ,t,φ), Rr(M,θ,t,φ))
# # include("C:/Users/mantas/AppData/Local/Temp/jl_56RGhMZEBm.jl")
# function dPr_dt!(dPr,Pr,(M,θ,φ),t)#(M, out, θ, t, φ, Pr)
#     riccati!(dPr,Ar(M,θ,t,φ),Kr(M,θ,t,φ,Pr),Pr,Qr(M,θ,t,φ),Rr(M,θ,t,φ))
# end
# # Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_dt!(M,dPr,θ,t,φ(t),Pr)

# Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_auto(dPr, Ar(M, θ, t, φ(t)), Br(M, θ, t, φ(t)), Pr, Qr(M, θ, t, φ(t)), Rr(M, θ, t, φ(t)))
# # ----------------------------------- 4. ode equations ----------------------------------- #
# function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ; tol = 1e-5, maxiters = 20) where {NX,NU,NΘ}
#     Pr_f = diagm(ones(NX))
#     Pr = ODE(dPr_dt!, Pr_f, (tf,t0), (M,θ,φ), Buffer{Tuple{NX,NX}}())
#     ξ = ODE(ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))

# end



# ----------------------------------- 5. ode solutions ----------------------------------- #
# include("odes.jl")

# ----------------------------------- 6. trajectories ----------------------------------- #


# ----------------------------------- PRONTO loop ----------------------------------- #

# ----------------------------------- guess functions ----------------------------------- #

end