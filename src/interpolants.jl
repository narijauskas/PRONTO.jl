# a quick and dirty linear matrix interpolant by way of wrapping SciMLBase
#TODO: replace this with a simple n-dim ArrayInterpolation/Timeseries object, which interpolates elementwise
# this is valid for state vectors/arrays, not for arbitrary matrices

import SciMLBase
import SciMLBase: LinearInterpolation
using StaticArrays


struct Interpolant{T}
    itp::LinearInterpolation{Vector{Float64}, Vector{T}}
end


# make callable as X(t)
(X::Interpolant{T})(tvals) where {T} = SciMLBase.interpolation(tvals,X.itp,nothing,Val{0},nothing,:left)::T
(X::Interpolant)(val,tvals) = SciMLBase.interpolation!(val,tvals,X.itp,nothing,Val{0},nothing,:left)

# where f is a function f(t)
function Interpolant(f::Function,ts,dims...;TT=Float64)
    S = Tuple{dims...}
    xs = map(t->MArray{S,TT}(f(t)),ts)
    T = eltype(xs)
    itp = LinearInterpolation(collect(ts), xs)
    return Interpolant{T}(itp)
end

# pre-allocate zeros if no function provided
Interpolant(ts,dims...;kw...) = Interpolant(t->zeros(dims...), ts, dims...; kw...)

# indexable
Base.firstindex(X::Interpolant) = 1
Base.lastindex(X::Interpolant) = length(X)
Base.length(X::Interpolant) = length(X.itp.t)
Base.getindex(X::Interpolant, inds...) = getindex(X.itp.u,inds...)
Base.setindex!(X::Interpolant, val, inds...) = setindex!(X.itp.u, val, inds...)

# iterable
Base.iterate(X::Interpolant, i=1) = i > length(X) ? nothing : (X[i], i+1)

# other functionality 
Base.show(io::IO, ::Interpolant{T}) where {T} = print(io, "Interpolant of $T")
Base.eltype(::Interpolant{T}) where {T} = T
# ideally, T is a MMatrix{2,2,F64} or something like that

times(X::Interpolant) = X.itp.t



function update!(f,X::Interpolant{T}) where {T}
    for(Xi, t) in zip(X,times(X))
        copy!(Xi, f(t))
    end
end

# update each X(t) by re-solving the ode from x0
function update!(X::Interpolant, ode, x0)
    reinit!(ode,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(ode, X.itp.t))
        X[i] .= x
        #TEST:
        # map!(v->v, X[i], x)
    end
    return nothing
end



#=
#MAYBE: interpolant knows element size/type
#MAYBE: just write simple custom interpolant?

#MAYBE:
to allow more interpolant types in the future:
    import SciMLBase: AbstractDiffEqInterpolation as AbstractInterpolation
    import SciMLBase: LinearInterpolation, ConstantInterpolation
or possibly:
    abstract type AbstractInterpolation end

#MAYBE:
alternatively write simple, custom linear interpolation type
=#













#=

splitseries(X) = SplitSeries(X)

struct SplitSeries
    X::Interpolant
end

function Base.iterate(sp::SplitSeries, i=1)
    i > length(eltype(sp.X)) ? nothing : begin
        x = collect(sp.X(t)[i] for t in times(sp.X))
        t = times(sp.X)
        return ((x,t),i+1)
    end
end

=#
