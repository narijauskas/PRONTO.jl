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
import LinearAlgebra: mul!, I, norm
using MatrixEquations # provides arec

using OrdinaryDiffEq
using DiffEqCallbacks

using Symbolics # code generation
import Symbolics: derivative # code generation
using SymbolicUtils.Code #MAYBE: code generation
export Num # this reexport is needed for codegen macro scoping

using ThreadTools # tmap for code generation
using UnicodePlots # data representation


using MacroTools
import MacroTools: @capture


using Base: OneTo
using Base: fieldindex
import Base: extrema, length, eachindex, show, size, eltype, getproperty, getindex


export @define_f, @define_l, @define_m, @define_Qr, @define_Rr
export @define_Q, @define_R
export @dynamics, @incremental_cost, @terminal_cost, @regulator_Q, @regulator_R
export resolve_model, symbolic
export zero_input, open_loop, closed_loop, smooth
export pronto
export Trajectory
export SymModel
export Model
export nx,nu,nθ



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

info(str) = println(as_tag(crayon"magenta","PRONTO"), str)
info(i, str) = println(as_tag(crayon"magenta","PRONTO[$i]"), str)
iinfo(str) = println("    > ", str) # secondary-level
iiinfo(str) = println("        > ", str) # tertiary-level


# ----------------------------------- # model functions ----------------------------------- #

#MAYBE: just throw a method error?
struct ModelDefError <: Exception
    θ::Model
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.θ)
    print(io, "PRONTO is missing method definitions for the $T model.\n")
end


# user can override this for some quantum-specific problems
γmax(θ::Model, ζ, τ) = 1.0
#TODO: sphere()
sphere(r, ζ, τ) = sqrt(r)/maximum(norm(ζ.x(t)) for t in LinRange(τ..., 10000))
# user can override this to change the preview displayed on each iteration
preview(θ::Model, ξ) = ξ.x

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
include("search_direction.jl") # lagrangian, 1st/2nd order optimizer, search direction
include("cost.jl") # cost and cost derivatives
include("armijo.jl") # armijo step and projection


struct Data
    ξ::Vector{Trajectory}
    Kr::Vector{Regulator}
    λ::Vector{ODE}
    Ko::Vector{OptFBGain}
    vo::Vector{OptFFWInput}
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
    OptFBGain[],
    OptFFWInput[],
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
                tol = 1e-6, 
                maxiters = 100,
                show_info = true,
                show_preview = true,
                show_steps = false,
                resample_dt = 0.001,
                armijo_kw...)
                
    solve_start = time_ns()
    show_info && info(0, "starting PRONTO")
    data = Data()

    for i in 1:maxiters
        loop_start = time_ns()
        push!(data.ξ, ξ)

        # -------------- build regulator -------------- #
        # α,μ -> Kr,x,u
        show_steps && iinfo("regulator")
        Kr = regulator(θ, ξ, τ)
        push!(data.Kr, Kr)
        # ξ = projection(θ, x0, φ, Kr, τ; verbosity)

        # -------------- search direction -------------- #
        # Kr,x,u -> z,v
        show_steps && iinfo("lagrange multipliers")
        λ = lagrange(θ, ξ, Kr, τ)
        push!(data.λ, λ)
        
        show_steps && iinfo("optimal feedback gain")
        Ko = opt_fb_gain(θ, λ, ξ, τ)
        push!(data.Ko, Ko)
        show_steps && iinfo("using $(is2ndorder(Ko) ? "2nd" : "1st") order search")

        show_steps && iinfo("optimal feedforward input")
        vo = opt_ffw_input(θ, λ, ξ, Ko, τ)
        push!(data.vo, vo)

        show_steps && iinfo("search direction")
        ζ = search_direction(θ, ξ, Ko, vo, τ; resample_dt)
        push!(data.ζ, ζ)

        
        # -------------- cost/derivatives -------------- #
        show_steps && iinfo("cost derivatives")

        h = cost(ξ, τ)
        push!(data.h, h)

        Dh,D2g = cost_derivs(θ, λ, ξ, ζ, τ)
        push!(data.Dh, Dh)
        push!(data.D2g, D2g)

        if Dh > 0
            solve_time = (time_ns() - solve_start)/1e9
            show_info && info(i, @sprintf("increased cost in %.2f seconds - quitting", solve_time))
            return ξ,data
        end

        # -------------- select γ via armijo step -------------- #
        show_steps && iinfo("armijo backstep")
        φ,γ = armijo(θ, x0, ξ, ζ, Kr, h, Dh, τ; resample_dt, armijo_kw...)
        push!(data.γ, γ)
        push!(data.φ, φ) 

        # -------------- runtime info -------------- #
        loop_time = (time_ns() - loop_start)/1e6
        infostr = @sprintf("Dh = %.3e, h = %.3e, γ = %.3e, ", Dh, h, γ)
        infostr *= ", order = $(is2ndorder(Ko) ? "2nd" : "1st"), "
        infostr *= @sprintf("solved in %.3f ms", loop_time)
        show_info && info(i, infostr)
        show_preview && plot_preview(θ, ξ)

        # -------------- check convergence -------------- #
        if -Dh < tol
            solve_time = (time_ns() - solve_start)/1e9
            show_info && info(i, @sprintf("PRONTO converged in %.2f seconds", solve_time))
            return φ,data
        end

        # -------------- update trajectory -------------- #
        ξ = φ # ξ_k+1 = φ_k
    end
    solve_time = (time_ns() - solve_start)/1e9
    show_info && info(maxiters, @sprintf("maxiters reached in %.2f seconds - quitting", solve_time))
    return ξ,data
end

end # module