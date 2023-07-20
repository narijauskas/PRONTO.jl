# PRONTO.jl v1.0.0_dev
module PRONTO

using Crayons
using FunctionWrappers
import FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures #TODO: test speed
using Base: @kwdef
using Printf

using Interpolations # data storage/resampling

import LinearAlgebra # collision with lu!
import LinearAlgebra: mul!, I
using MatrixEquations # provides arec

using OrdinaryDiffEq
using DiffEqCallbacks

using Symbolics # code generation
import Symbolics: derivative # code generation
using SymbolicUtils.Code #MAYBE: code generation
export Num # this reexport is needed for codegen macro scoping

using ThreadTools # tmap for code generation
using UnicodePlots # data representation

using Dates: now

using MacroTools
import MacroTools: @capture


using Base: OneTo
using Base: fieldindex
import Base: extrema, length, eachindex, show, size, eltype, getproperty, getindex


export @define_f, @define_l, @define_m, @define_Q, @define_R
export @dynamics, @incremental_cost, @terminal_cost, @regulator_Q, @regulator_R
export resolve_model, symbolic
export zero_input, open_loop, closed_loop, smooth
export pronto
export Trajectory
export SymModel
export Model
export nx,nu,nθ



#TODO: reconsider export
export ODE, Buffer
export preview


# ----------------------------------- model type ----------------------------------- #


abstract type Model{NX,NU} end
# fields can be Scalars or SArrays

nx(::Model{NX,NU}) where {NX,NU} = NX
nu(::Model{NX,NU}) where {NX,NU} = NU
nθ(::T) where {T<:Model} = nθ(T)

nx(::Type{<:Model{NX,NU}}) where {NX,NU} = NX
nu(::Type{<:Model{NX,NU}}) where {NX,NU} = NU
nθ(T::Type{<:Model}) = fieldcount(T)

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



# ----------------------------------- #. helpers ----------------------------------- #
# include("helpers.jl") #TODO: remove file

as_tag(str) = as_tag(crayon"default", str)
as_tag(c::Crayon, str) = as_color(c, as_bold("[$str: "))
as_color(c::Crayon, str) = "$c" * str * "$(crayon"default")"
as_bold(ex) = as_bold(string(ex))
as_bold(str::String) = "$(crayon"bold")" * str * "$(crayon"!bold")"
clearln() = print("\e[2K","\e[1G")

info(str; verbosity=1) = verbosity >= 1 && println(as_tag(crayon"magenta","PRONTO"), str)
info(i, str; verbosity=1) = verbosity >= 1 && println(as_tag(crayon"magenta","PRONTO[$i]"), str)
iinfo(str; verbosity=2) = verbosity >= 2 && println("    > ", str) # secondary-level
iiinfo(str; verbosity=3) = verbosity >= 3 && println("        > ", str) # tertiary-level


# ----------------------------------- #. model functions ----------------------------------- #

#MAYBE: just throw a method error?
struct ModelDefError <: Exception
    θ::Model
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.θ)
    print(io, "PRONTO is missing method definitions for the $T model.\n")
end


# TODO: γlimit
# must result in γ∈[β^maxiters,1]

runtime_info(θ::Model, ξ; verbosity) = nothing



# by default, this is the solution to the algebraic riccati equation at tf
# user can override this behavior for a model type by defining PRONTO.Pf(θ::MyModel,αf,μf,tf)
function Pf(θ::Model{NX},αf,μf,tf) where {NX}
    Ar = fx(θ, αf, μf, tf)
    Br = fu(θ, αf, μf, tf)
    Qr = Q(θ, αf, μf, tf)
    Rr = R(θ, αf, μf, tf)
    Pf,_ = arec(Ar,Br*(Rr\Br'),Qr)
    return SMatrix{NX,NX,Float64}(Pf)
end



# ----------------------------------- #. components ----------------------------------- #

include("kernels.jl") # placeholder solver kernel function definitions
include("codegen.jl") # takes derivatives, generates model functions
include("odes.jl") # wrappers for ODE solution handling
include("regulator.jl") # regulator for projection operator
include("projection.jl") # projected (closed loop) and guess (open loop) trajectories
include("optimizer.jl") # lagrangian, 1st/2nd order optimizer, search direction
include("cost.jl") # cost and cost derivatives
include("armijo.jl") # armijo step and projection


struct Data
    ξ::Vector{Trajectory}
    Kr::Vector{Regulator}
    λ::Vector{ODE}
    Ko::Vector{Optimizer}
    vo::Vector{Costate}
    ζ::Vector{Trajectory}
    h::Vector{Float64}
    Dh::Vector{Float64}
    D2g::Vector{Float64}
    γ::Vector{Float64}
    φ::Vector{Trajectory}
end

Data() = Data(
    Trajectory[],
    Regulator[],
    ODE[],
    Optimizer[],
    Costate[],
    Trajectory[],
    Float64[],
    Float64[],
    Float64[],
    Float64[],
    Trajectory[],
)

Base.show(io::IO, data::Data) = print(io, "PRONTO data: $(length(data.φ)) iterations")
# ----------------------------------- guess trajectories ----------------------------------- #
# generate ξ0 from either η
function zero_input(θ::Model{NX,NU}, x0, τ) where {NX,NU}
    μ = t -> zeros(SVector{NU})
    open_loop(θ, x0, μ, τ)
end


function open_loop(θ::Model{NX,NU}, x0, μ, τ) where {NX,NU}
    α = t -> zeros(SVector{NX})
    Kr = (α,μ,t) -> zeros(SMatrix{NU,NX})
    projection(θ, x0, α, μ, Kr, τ)
end

#TODO: smooth guess
# smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
smooth(θ, x0, xf, τ) = smooth(θ, x0, xf, t->zeros(nu(θ)), τ)
function smooth(θ, x0, xf, μ, τ)
    t0,tf = τ
    α = t-> @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
    Kr = regulator(θ, α, μ, τ)
    projection(θ, x0, α, μ, Kr, τ)
end

function closed_loop(θ, x0, α, μ, τ)
    Kr = regulator(θ, α, μ, τ)
    projection(θ, x0, α, μ, Kr, τ)
end

# ----------------------------------- pronto loop ----------------------------------- #

fwd(τ) = extrema(τ)
bkwd(τ) = reverse(fwd(τ))


# solves for x(t),u(t)'
function pronto(θ::Model, x0::StaticVector, ξ::Trajectory, τ; 
                limitγ=false, 
                tol = 1e-6, 
                maxiters = 100,
                armijo_maxiters = 25,
                verbosity = 1,
                show_preview = true,
                )
    t0,tf = τ
    info(0, "starting PRONTO")
    data = Data()

    for i in 1:maxiters
        loop_start = time_ns()
        push!(data.ξ, ξ)

        # -------------- build regulator -------------- #
        # α,μ -> Kr,x,u
        Kr = regulator(θ, ξ, τ; verbosity)
        push!(data.Kr, Kr)
        # ξ = projection(θ, x0, φ, Kr, τ; verbosity)

        # -------------- search direction -------------- #
        # Kr,x,u -> z,v
        λ = lagrangian(θ, ξ, Kr, τ; verbosity)
        push!(data.λ, λ)
        
        Ko = optimizer(θ, λ, ξ, τ; verbosity)
        push!(data.Ko, Ko)

        vo = costate(θ, λ, ξ, Ko, τ; verbosity)
        push!(data.vo, vo)

        # λ,Ko,vo = optimizer(θ,ξ,Kr,τ; verbosity)
        ζ = search_direction(θ,ξ,Ko,vo,τ; verbosity)
        push!(data.ζ, ζ)

        
        # -------------- cost/derivatives -------------- #
        h = cost(ξ, τ)
        push!(data.h, h)

        Dh,D2g = cost_derivs(θ, λ, ξ, ζ, τ; verbosity)
        push!(data.Dh, Dh)
        push!(data.D2g, D2g)

        Dh > 0 && (info(i, "increased cost - quitting"); (return ξ,data))

        # -------------- select γ via armijo step -------------- #
        φ,γ = armijo(θ, x0, ξ, ζ, Kr, h, Dh, τ; verbosity, armijo_maxiters)
        push!(data.γ, γ)
        push!(data.φ, φ) 
        -Dh < tol && (info(i, as_bold("PRONTO converged")); (return φ,data))
        ξ = φ # ξ_k+1 = φ_k

        # -------------- runtime info -------------- #
        loop_time = (time_ns() - loop_start)/1e6
        infostr = @sprintf("Dh = %.3e, h = %.3e, γ = %.3e, ", Dh, h, γ)
        infostr *= ", order = $(is2ndorder(Ko) ? "2nd" : "1st"), "
        infostr *= @sprintf("solved in %.4f ms", loop_time)
        info(i, infostr; verbosity)
        show_preview && plot_preview(θ, ξ)
        
    end
    info(maxiters, "maxiters reached - quitting")
    return ξ,data
end

# γ = γmax; 
# α=0.4; β=0.7
# γmin = β^armijo_maxiters
# γ = limitγ ? min(1, 1/maximum(maximum(ζ.x(t) for t in t0:0.0001:tf))) : 1.0

# local η # defined to exist outside of while loop
# while γ > γmin
#     iiinfo("armijo γ = $(round(γ; digits=6))"; verbosity)
#     φ = armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)
#     g = cost(φ, τ)
#     h-g >= -α*γ*Dh ? break : (γ *= β)
# end

end # module