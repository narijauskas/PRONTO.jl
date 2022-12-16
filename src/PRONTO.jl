# PRONTO.jl dev_0.4
module PRONTO

using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
export @closure

using Base: @kwdef
export @kwdef
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

using Dates: now

using MacroTools
using MacroTools: @capture

using Interpolations

using Base: OneTo
using Base: fieldindex
import Base: extrema, length, eachindex, show, size, eltype, getproperty, getindex






export pronto
export info

export @tick,@tock,@clock

export ODE, Buffer
export preview

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


show(io::IO, ::T) where {T<:Model} = print(io, "$T model")

iscompact(io) = get(io, :compact, false)
function show(io::IO,::MIME"text/plain", θ::T) where {T<:Model}
    if iscompact(io)
        print(io, "$T model")
    else
        println(io, "$(as_bold(T)) model with parameter values:")
        for name in fieldnames(T)
            println(io, "  $name: $(getfield(θ,name))")
        end
    end
end


# typelength(::Type{<:Number}) = 1
# typelength(::Type{T}) where {T} = length(T)

# function Base.iterate(θ::Model, state=1)
#     # go thru each the length of each type
#     # if bigger than state

#     (value, state+1)
# end
# function getindex(θ::Model, i)
#     iterate thru until value
# end




struct SymbolicModel{T}
    vars
end

function SymbolicModel(T::DataType)
    @variables θ[1:nθ(T)]
    SymbolicModel{T}(collect(θ))
end
# getproperty(θ::SymbolicModel{T}, name::Symbol) where {T} = getindex(θ.vars, fieldindex(T, name))
getindex(θ::SymbolicModel{T}, i::Integer) where {T} = getindex(getfield(θ, :vars), i)
getproperty(θ::SymbolicModel{T}, name::Symbol) where {T} = getindex(θ, fieldindex(T, name))


# ----------------------------------- #. helpers ----------------------------------- #
include("helpers.jl")


#YO: can I actually deprecate this? :)
views(::Model{NX,NU,NΘ},ξ) where {NX,NU,NΘ} = (@view ξ[1:NX]),(@view ξ[NX+1:end])


# ----------------------------------- #. model functions ----------------------------------- #
# definitions for these must be generated from a user-specified model by codegen


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
function Pf(α,μ,tf,θ::Model{NX}) where {NX}
    # xref = θ.ref
    # uref = @SVector zeros(nu(θ))
    Ar = fx(α, μ, tf, θ)
    Br = fu(α, μ, tf, θ)
    Qr = Q(α, μ, tf, θ)
    Rr = R(α, μ, tf, θ)
    Pf,_ = arec(Ar,Br*(Rr\Br'),Qr)
    # Pf,_ = ared(Ar,Br,Rr,Qr)
    return SMatrix{NX,NX,Float64}(Pf)
end

#     B*inv(R)*B'
#    # solve algebraic riccati eq at time T to get terminal cost
#    Pt,_ = arec(A(ξ,T), B(ξ,T)inv(Rr(T))B(ξ,T)', Qr(T))
   

# solution to DARE at desired equilibrium
# ref = zeros(NX)
# Pp,_ = ared(fx(ref,zeros(NU)), fu(ref,zeros(NU)), Rlqr, Qlqr)

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



# ----------------------------------- #. components ----------------------------------- #


include("codegen.jl") # takes derivatives, generates model functions
include("odes.jl") # ODE solution handling
include("regulator.jl")
include("projection.jl")
include("optimizer.jl")
include("armijo.jl")



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
function pronto(θ::Model{NX,NU,NΘ}, x0::StaticVector, φ, τ; limitγ=false, tol = 1e-5, maxiters = 20,verbose=true) where {NX,NU,NΘ}
    t0,tf = τ

    for i in 1:maxiters
        # info(i, "iteration")
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
        # print(ξ)

        # -------------- select γ -------------- #

        # γ = γmax; 
        aα=0.4; aβ=0.7
        γ = limitγ ? min(1, 1/maximum(maximum(ζ.x(t) for t in t0:0.0001:tf))) : 1.0

        local η
        while γ > aβ^25
            verbose && iinfo("armijo γ = $(round(γ; digits=6))")
            η = armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)
            g = cost(η, τ)
            h-g >= -aα*γ*Dh ? break : (γ *= aβ)
        end
        verbose && info(i, "Dh = $Dh, h = $h, γ=$γ") #TODO: first/second order

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


end # module