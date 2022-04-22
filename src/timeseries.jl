using DataInterpolations
# a timeseries U(t) is a tensor-valued time-varying interpolant
# U(t) -> u, where u is a tensor (represented by Array)


struct Timeseries{T}
    u::DataInterpolations.AbstractInterpolation
    size::NTuple
end

# constructor from mapping
# t->f(t) ∀ t ∈ τ
# Timeseries(t->f(t), τ)

function Timeseries(f, τ)
    u = LinearInterpolation(map(f, τ), τ)
    u0 = u(first(τ)); T = typeof(u0); sz = size(u0)
    return Timeseries{T}(u, sz)
end

# Interpolation
# U(t) -> u (of type T)
(U::Timeseries{T})(t) where {T} = U.u(t)


Base.size(U::Timeseries) = U.size
# Base.show(io::IO, U::Timeseries)




# maybe:
# φ[1](t) # fundamentally inefficient with the current implementation
# Base.getindex(φ::Trajectory, i::Int) = 1 <= i <= length(φ) ? (return t->φ(t)[i]) : throw(BoundsError(φ, i))
# for v in φ; v(t); end
# Base.iterate(φ::Trajectory, state=1) = state <= length(φ) ? (return (φ[state], state+1)) : (return nothing)

