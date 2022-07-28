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

function findindex(t,τ)
    tdir = sign(t[end] - t[1])
    tdir * τ > tdir * t[end] && error("Solution interpolation cannot extrapolate past the final timepoint.")
    tdir * τ < tdir * t[1] && error("Solution interpolation cannot extrapolate before the first timepoint.")
    @inbounds i = searchsortedfirst(t, τ, rev=tdir < 0) # It's in the interval t[i-1] to t[i]
    return i
end

# set X to be x(τ) from set x and t
function interpolate!(X,τ,x,t)
    i = findindex(t,τ)

    @inbounds if t[i] == τ
        copy!(X, x[i])
    elseif t[i-1] == τ # Can happen if it's the first value!
        copy!(X, x[i-1])
    else
        dt = t[i] - t[i-1]
        Θ = (τ - t[i-1]) / dt
        Θm1 = (1 - Θ)
        @. X = Θm1 * x[i-1] + Θ * x[i]
    end
    return X
end


# pure update, 29 μs, zero allocations
# evals push into ms range, MiB of allocations
function update!(X::Interpolant, f)
    for (x,t) in zip(X.x,X.t)
        x .= f(t)
    end
end
# foreach((x,t)->(x .= f(t)), zip(X.x,X.t))


# indexable
Base.firstindex(X::Interpolant) = 1
Base.lastindex(X::Interpolant) = length(X)
Base.length(X::Interpolant) = length(X.t)
Base.getindex(X::Interpolant, inds...) = getindex(X.x,inds...)
Base.setindex!(X::Interpolant, val, inds...) = setindex!(X.x, val, inds...)

# iterable
Base.iterate(X::Interpolant, i=1) = i > length(X) ? nothing : (X[i], i+1)

# show
Base.show(io::IO, ::Interpolant{S,<:Any,<:Any}) where {S} = print(io, "$S - element Interpolant")

# times(X::Interpolant) = X.t
# Base.eltype(::Interpolant{T}) where {T} = T




# ----------------------------------- Trajectories ----------------------------------- #

#=
struct Trajectory{SX,SU,A,T}
    x::Vector{MVector{SX,A}}
    u::Vector{MVector{SU,A}}
    t::T
    X::MVector{SX,A}
    U::MVector{SU,A}
end

(Ξ::Trajectory)(τ) = interpolate!(Ξ.X, Ξ.U, τ, Ξ.x, Ξ.u, Ξ.t)

function Trajectory(X::Interpolant, U::Interpolant, t)
    @assert X.t == U.t == t "unequal timesteps"
    Trajectory(X.x, X.u, t, X.X, U.X)
end

function interpolate!(X,U,τ,x,u,t)
    i = find_index(t,τ)

    @inbounds if t[i] == τ
        copy!(X, x[i])
        copy!(U, u[i])
    elseif t[i-1] == τ # Can happen if it's the first value!
        copy!(X, x[i-1])
        copy!(U, u[i-1])
    else
        dt = t[i] - t[i-1]
        Θ = (τ - t[i-1]) / dt
        Θm1 = (1 - Θ)
        @. X = Θm1 * x[i-1] + Θ * x[i]
        @. U = Θm1 * u[i-1] + Θ * u[i]
    end
    return (X,U)
end

=#