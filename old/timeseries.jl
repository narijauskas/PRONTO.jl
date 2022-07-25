
using SciMLBase

abstract type Timeseries{T} end


struct InterpolatedTimeseries{T} <: Timeseries{T}
    interp::SciMLBase.AbstractDiffEqInterpolation
    p::Any
    tspan::Tuple{Float64,Float64}
end


#= extended functionality
function (X::InterpolatedTimeseries{T})(t,::Type{deriv}=Val{0}; idxs=nothing, continuity=:left) where {T, deriv}
    X.interp(t, idxs, deriv, X.p, continuity)
end
=#

function (X::InterpolatedTimeseries{T})(t; idxs=nothing, continuity=:left)::T where {T}
    X.interp(t, idxs, Val{0}, X.p, continuity)
end

# wraps an ODESolution
function Timeseries(sol::SciMLBase.AbstractODESolution)
    InterpolatedTimeseries{typeof(first(sol))}(sol.interp, sol.prob.p, sol.prob.tspan)
end

# samples ∀ t ∈ τ, creating a linear interpolation
function Timeseries(f::Function, τ)
    interp = SciMLBase.LinearInterpolation(τ, map(f,τ))
    tspan = (first(τ), last(τ))
    InterpolatedTimeseries{typeof(f(first(τ)))}(interp, nothing, tspan)
end


Base.show(io::IO, ::Timeseries{T}) where {T} = print(io, "Timeseries{$T}")


struct FunctionalTimeseries{T} <: Timeseries{T}
    f::Function
end

# just holds the function and it's return type
function Timeseries(f::Function; t₀=0.0)
    FunctionalTimeseries{typeof(f(t₀))}(f)
end

# allows X(t) -> returns type T
function (X::FunctionalTimeseries{T})(t)::T where {T}
    X.f(t)
end











# using DataInterpolations
# # a timeseries U(t) is a tensor-valued time-varying interpolant
# # U(t) -> u, where u is a tensor (represented by Array)


# struct Timeseries{T}
#     u::DataInterpolations.AbstractInterpolation
#     size::NTuple
# end

# # constructor from mapping
# # t->f(t) ∀ t ∈ τ
# # Timeseries(t->f(t), τ)

# function Timeseries(f, τ)
#     u = LinearInterpolation(map(f, τ), τ)
#     u0 = u(first(τ)); T = typeof(u0); sz = size(u0)
#     return Timeseries{T}(u, sz)
# end

# # Interpolation
# # U(t) -> u (of type T)
# (U::Timeseries{T})(t) where {T} = U.u(t)
# #TODO: benchmark with and without type annotation
# # (U::Timeseries{T})(t) where {T} = U.u(t)::T


# Base.size(U::Timeseries) = U.size
# # Base.show(io::IO, U::Timeseries)




# maybe:
# φ[1](t) # fundamentally inefficient with the current implementation
# Base.getindex(φ::Trajectory, i::Int) = 1 <= i <= length(φ) ? (return t->φ(t)[i]) : throw(BoundsError(φ, i))
# for v in φ; v(t); end
# Base.iterate(φ::Trajectory, state=1) = state <= length(φ) ? (return (φ[state], state+1)) : (return nothing)

