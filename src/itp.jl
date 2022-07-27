# adapted from SciMLBase
# simplified, non-allocating

using StaticArrays

struct LinearInterpolation{S,T,TT}
    x::Vector{MVector{S,T}}
    t::TT
    buf::MVector{S,T}
end

# where f is a function f(t)
function LinearInterpolation(f::Function,ts::TT;T=Float64) where {TT}
    S = length(f(first(ts)))
    xs = map(t->MVector{S,T}(f(t)), ts)
    buf = MVector{S,T}(undef)
    LinearInterpolation{S,T,TT}(xs,ts,buf)
end


function (X::LinearInterpolation{S,T,TT})(τ)::MVector{S,T} where {S,T,TT}
    t = X.t
    x = X.x
    buf = X.buf

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
        Θ = (tval - t[i-1]) / dt
        Θm1 = (1 - Θ)
        @. buf = Θm1 * x[i-1] + Θ * x[i]
    end
    return buf
end

# LinearInterpolation(t -> sin(t), ts)



ts = 0:0.001:10
itp = LinearInterpolation(t->([1,2,3,4]*sin(t)), ts)
