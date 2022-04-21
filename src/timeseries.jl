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


Base.size(U::Timeseries) = u.size
# Base.show()


# dims