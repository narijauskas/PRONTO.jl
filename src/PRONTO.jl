module PRONTO

using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
export @closure
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

# S is a Tuple{dims...}
Buffer{S} = MArray{S, Float64}
Buffer{S}() where {S} = zeros(MArray{S, Float64})



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
# defines all PRONTO.f(M,θ,t,ξ) and PRONTO.f!(M,θ,t,ξ) along with derivatives
include("model.jl")



# ----------------------------------- 3. intermediate operators ----------------------------------- #

function Ar(M::Model{NX,NU,NΘ},θ,t,φ) where {NX,NU,NΘ}
    buf = Buffer{Tuple{NX,NX}}()
    fx!(M,buf,θ,t,φ)
    return buf
end


# ----------------------------------- 4. ode equations ----------------------------------- #

# ----------------------------------- 5. ode solutions ----------------------------------- #
include("odes.jl")

# ----------------------------------- 6. trajectories ----------------------------------- #



# ----------------------------------- * buffer type ----------------------------------- #
# ----------------------------------- * timing ----------------------------------- #




# ----------------------------------- PRONTO loop ----------------------------------- #

# ----------------------------------- guess functions ----------------------------------- #
include("guess.jl")

end