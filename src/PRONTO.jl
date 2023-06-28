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


#TODO: #FUTURE: using OrdinaryDiffEq
# using DifferentialEquations
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


export pronto
export info #TODO: does this need to be exported?


#TODO: reconsider export
export ODE, Buffer
export preview

# ----------------------------------- model type ----------------------------------- #

export Model
export nx,nu,nθ


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
# user can override this behavior for a model type by defining PRONTO.Pf(α,μ,tf,θ::MyModel)
function Pf(θ::Model{NX},α,μ,tf) where {NX}
    Ar = fx(θ, α, μ, tf)
    Br = fu(θ, α, μ, tf)
    Qr = Q(θ, α, μ, tf)
    Rr = R(θ, α, μ, tf)
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



# ----------------------------------- pronto loop ----------------------------------- #

fwd(τ) = extrema(τ)
bkwd(τ) = reverse(fwd(τ))

# preview(θ::Model, ξ) = nothing

# solves for x(t),u(t)'
function pronto(θ::Model, x0::StaticVector, φ, τ; 
                limitγ=false, 
                tol = 1e-6, 
                maxiters = 20,
                armijo_maxiters = 25,
                verbosity = 1,
                verbose = false,
                )
    t0,tf = τ
    # verbose && 
    info(0, "starting PRONTO")

    for i in 1:maxiters
        # info(i, "iteration")
        # -------------- build regulator -------------- #
        # α,μ -> Kr,x,u
        Kr = regulator(θ, φ, τ; verbosity)
        ξ = projection(θ, x0, φ, Kr, τ; verbosity)

        # -------------- search direction -------------- #
        # Kr,x,u -> z,v
        λ = lagrangian(θ,ξ,φ,Kr,τ; verbosity)
        Ko = optimizer(θ,λ,ξ,φ,τ; verbosity)
        vo = costate(θ,λ,ξ,φ,Ko,τ; verbosity)
        ζ = search_direction(θ,ξ,Ko,vo,τ; verbosity)

        # -------------- cost/derivatives -------------- #
        h = cost(ξ, τ)
        Dh,D2g = cost_derivs(θ,λ,φ,ξ,ζ,τ; verbosity)
        Dh > 0 && (info(i, "increased cost - quitting"); (return φ))
        -Dh < tol && (info(i-1, as_bold("PRONTO converged")); (return φ))


        # -------------- select γ via armijo step -------------- #
        # γ = γmax; 
        α=0.4; β=0.7
        γmin = β^armijo_maxiters
        γ = limitγ ? min(1, 1/maximum(maximum(ζ.x(t) for t in t0:0.0001:tf))) : 1.0

        local η # defined to exist outside of while loop
        while γ > γmin
            iiinfo("armijo γ = $(round(γ; digits=6))"; verbosity)
            η = armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)
            g = cost(η, τ)
            h-g >= -α*γ*Dh ? break : (γ *= β)
        end
        φ = η

        #TODO: store intermediates Kr,ξ,λ,Ko,vo,ζ,h,Dh,D2g,γ,        
        infostr = @sprintf("Dh = %.3e, h = %.3e, γ = %.3e, ", Dh, h, γ)
        infostr *= ", order = $(is2ndorder(Ko) ? "2nd" : "1st")"
        # info(i, "Dh = $Dh, h = $h, γ = $γ, order = $(is2ndorder(Ko) ? "2nd" : "1st")"; verbosity)
        info(i, infostr; verbosity)
        runtime_info(θ, ξ; verbosity)
        # println(preview(φ.x, 1))
        # println(preview(ξ.x, (1,3)))
        # println(preview(ξ.x, (2,4)))
        # println(preview(ξ.u, 1))
    end
    return φ
end


end # module