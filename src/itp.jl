# adapted from SciMLBase (some parts copied directly)
# simplified, non-allocating
# long term goal: integrate, use non-linear interpolants, variable time step everywhere

# should be well suited for representing smooth, dense, continuous things like states, costates, and inputs
using StaticArrays

struct Interpolant{S,A,T}
    x::Vector{MVector{S,A}}
    t::T
    buf::MVector{S,A}
end

# where f is a function f(t)
function Interpolant(f::Function,ts::T; A=Float64) where {T}
    S = length(f(first(ts)))
    xs = map(t->MVector{S,A}(f(t)), ts)
    buf = MVector{S,A}(undef)
    Interpolant{S,A,T}(xs,ts,buf)
end

# where x is a matrix of size NX x length(ts)
# in other words, column vectors of x
function Interpolant(x::AbstractMatrix, ts::T,dims...; A=Float64) where {T}
    S,l = size(x)
    @assert l = length(ts) "time dimension mismatch"
    xs = map(col->MVector{S,A}(col), eachcol(x))
    buf = MVector{S,A}(undef)
    Interpolant{S,A,T}(xs,ts,buf)
end


function (X::Interpolant{S,A,T})(τ)::MVector{S,A} where {S,A,T}
    interpolate!(X.buf, τ, X.x, X.t)
end

# Interpolant(t -> sin(t), ts)

# set buf to be x(τ) from set x and t
function interpolate!(buf,τ,x,t)
    tdir = sign(t[end] - t[1])
    tdir * τ > tdir * t[end] && error("Solution interpolation cannot extrapolate past the final timepoint.")
    tdir * τ < tdir * t[1] && error("Solution interpolation cannot extrapolate before the first timepoint.")
    @inbounds i = searchsortedfirst(t, τ, rev=tdir < 0) # It's in the interval t[i-1] to t[i]
    @inbounds if t[i] == τ
        copy!(buf, x[i])
    elseif t[i-1] == τ # Can happen if it's the first value!
        copy!(buf, x[i-1])
    else
        dt = t[i] - t[i-1]
        Θ = (τ - t[i-1]) / dt
        Θm1 = (1 - Θ)
        @. buf = Θm1 * x[i-1] + Θ * x[i]
    end
    return buf
end


# pure update, 29 μs, zero allocations
# evals push into ms range, MiB of allocations
function update!(X::Interpolant, ode)
    for (x,t) in zip(X.x,X.t)
        # x .= f(t)
        ode(x,t)
        # src!(x, t) in-place version?
    end
end

Base.length(X::Interpolant) = length(X.t)
Base.show(io::IO, ::Interpolant{S,<:Any,<:Any}) where {S} = print(io, "$S - element Interpolant")

# times(X::Interpolant) = X.t
# Base.eltype(::Interpolant{T}) where {T} = T




# ts = 0:0.001:10
# ts = collect(ts)
# itp = Interpolant(t->([1,2,3,4]*sin(t)), ts)

