#=
to allow more interpolant types in the future:
    import SciMLBase: AbstractDiffEqInterpolation as AbstractInterpolation
    import SciMLBase: LinearInterpolation, ConstantInterpolation
or possibly:
    abstract type AbstractInterpolation end
=#
import SciMLBase
import SciMLBase: LinearInterpolation
using StaticArrays

# Interpolant{T}(ts) where {T} = 

# function Interpolant{T}(ts) where {T}
#     # pre-allocate a matrix for each t in ts
#     itp = LinearInterpolation(ts, map(t->zeros(T),ts))
#     new{T}(itp)
# end    

struct Interpolant{T}
    itp::LinearInterpolation{Vector{Float64}, Vector{T}}
end

Base.eltype(::Interpolant{T}) where {T} = T
# ideally, T is a MMatrix{2,2,F64} or something like that

Base.show(io::IO, ::Interpolant{T}) where {T} = print(io, "Interpolant of $T")

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

Base.firstindex(X::Interpolant) = 1
Base.lastindex(X::Interpolant) = length(X)
Base.length(X::Interpolant) = length(X.itp.t)
Base.getindex(X::Interpolant, inds...) = getindex(X.itp.u,inds...)
Base.setindex!(X::Interpolant, val, inds...) = setindex!(X.itp.u, val, inds...)

function update!(f,X::Interpolant{T}) where {T}
    for (i,t) in enumerate(X.itp.t)
        # X[i] = f(t)
        map!(x->x, X[i], f(t))
    end
end















        # X[i] .= f(t)
        # map!(x->0,X[i],X[i])
        # map!(t->T(f(t)), X[i], Scalar(t))

    # map!(t->f(t), X.itp.u, X.itp.t)







# function Interpolant(f,ts)
#     # x0 = f(first(ts))
#     # T = SArray{Tuple{size(x0)...},eltype(x0)}
#     # LinearInterpolation(ts, map(t->zeros(T),ts))
#     # T = eltype(x0)
#     # LinearInterpolation(ts, map(t->ones(T,size(x0)...),ts))


#     LinearInterpolation(ts, map(t->f(t),ts))

#     # Interpolant{T}(ts)
#     # T = SMatrix{size(x0)...,eltype(x0)}
#     # x0 = @SArray f(first(ts))
#     # get T as StaticArray ...or MMatrix for inplace updates?
#     # Interpolant{typeof(x0)}(ts)
# end

# (X::LinearInterpolation)(tvals) = SciMLBase.interpolation(tvals,X,nothing,Val{0},nothing,:left)
# (X::LinearInterpolation)(val,tvals) = SciMLBase.interpolation!(val,tvals,X,nothing,Val{0},nothing,:left)

# Base.setindex!(X::LinearInterpolation, val, inds...) = setindex!(X.u, val, inds...)

# function update!(f, X)
#     for (i,t) in enumerate(X.t)
#         X[i] = f(t) # setindex!(X, f(t), i)
#         # SMatrix or MMatrix?
#     end
# end

#=
struct Interpolant{T}
    itp::LinearInterpolation

    function Interpolant{T}(ts) where {T}
        # pre-allocate a matrix for each t in ts
        itp = LinearInterpolation(ts, map(t->zeros(T),ts))
        new{T}(itp)
    end    
end


function Interpolant(f,ts)
    x0 = f(first(ts))
    T = SArray{Tuple{size(x0)...},eltype(x0)}
    Interpolant{T}(ts)
    # T = SMatrix{size(x0)...,eltype(x0)}
    # x0 = @SArray f(first(ts))
    # get T as StaticArray ...or MMatrix for inplace updates?
    # Interpolant{typeof(x0)}(ts)
end

# function Interpolant(f,t)
#     itp = LinearInterpolation(t, map(τ->f(τ),t))
#     T = typeof(first(itp.u))
#     return Interpolant{T}(itp)
# end

#getproperty returns union?
function Base.getproperty(x::Interpolant, name::Symbol)
    itp = getfield(x, :itp)
    T = fieldtype(typeof(itp), name)
    getproperty(itp, name)::T
end
Base.setindex!(X::Interpolant, val, inds...) = setindex!(X.u, val, inds...)
Base.eltype(::Type{Interpolant{T}}) where {T} = T



#X(t)
(X::Interpolant{T})(tvals) where {T} = SciMLBase.interpolation(tvals,getfield(X, :itp),nothing,Val{0},nothing,:left)::T
(X::Interpolant{T})(val,tvals) where {T} = SciMLBase.interpolation!(val,tvals,getfield(X, :itp),nothing,Val{0},nothing,:left)::T
# (X::Interpolant)(tvals) = SciMLBase.interpolation(tvals,X.itp,nothing,Val{0},nothing,:left)
# (X::Interpolant)(val,tvals) = SciMLBase.interpolation!(val,tvals,X.itp,nothing,Val{0},nothing,:left)

# X[i] = val

# update values of X by evaluating f(t) for each t
function update!(f, X::Interpolant{T}) where {T}
    for (i,t) in enumerate(X.t::AbstractVector{Number})
        X[i] = T(f(t)) # setindex!(X, f(t), i)
        # SMatrix or MMatrix?
    end
end

function update!(f, X)
    for (i,t) in enumerate(X.t)
        X[i] = f(t) # setindex!(X, f(t), i)
        # SMatrix or MMatrix?
    end
end

Base.length(X::Interpolant) = length(X.t)
Base.show(io::IO, X::Interpolant{T}) where {T} = print(io, "Interpolant{$T} on t∈$(extrema(X.t))")

#TODO: Interpolant{T} variants (may be much faster)

# Base.setindex!(X::Interpolant{T}, val, inds...) where {T} = setindex!(X.itp.u, val, inds...)::T


#MAYBE: to re-use time vector from interpolation:
# struct Interpolant{T}
#     itp::LinearInterpolation
#     t::Ref

#     function Interpolant(f,t)
#         itp = LinearInterpolation(t, map(τ->f(τ),t))
#         t = Ref(itp.t)
#         new(itp, t)
#     end
# end


=#

















#=TODO: clean up, re-add trajectory


# allows for x(t)

# defines the Trajectory type
# used to represent objects of the form ξ = (x(t),u(t))
# x and u are sampled to a fixed timestep
# based on a linear interpolant object (maybe more options in future)

# x and u are mutable, while the trajectory is not

# x and u are interpolants on StaticArrays? Arrays?
# have constant, known return types XT,UT?


#FIX: re-add trajectory type
struct Trajectory#{XT,UT}
    x::Interpolation # x::AbstractInterpolation
    u::Interpolation # u::AbstractInterpolation
    # t # t::Vector{Float64} ??
end

Trajectory(X,U,t) = Trajectory(
    Interpolation(t, map(τ->X(τ),t)),
    Interpolation(t, map(τ->U(τ),t))
)

#functionality provided by Interpolation
# ξ.x(t), ξ.u(t) 
# update!(U, ξ.u)
# update!(ξ.x) do t
#     X(t)
# end


# # evaluate X(t) for each t in ξ.t, store to ξ.x at t
# function update_x!(ξ, X)
#     # map!(t->X(t), ξ.x, ξ.t)
#     for (i,t) in enumerate(ξ.t)
#         setindex!(ξ.x, X(t), i)
#     end
# end

# function update_u!(ξ, U)
#     for (i,t) in enumerate(ξ.t)
#         setindex!(ξ.u, U(t), i)
#     end
# end


















# set_x!(ξ,i,val) vs set_x!(ξ,t,val)
# update the underlying data stored within the Interpolation
# updateindex!(x::T, val, i) where {T<:AbstractInterpolation} = @error "PRONTO does not support $T"
@inline function updateindex!(itp::LinearInterpolation, val, i)
    # check if t in itp.t ??
    itp.u[i] == val
end






# set the appropriate interpolant
function set_x!(ξ, t, val)
    updateindex!(ξ.x, val, )
end


function set_u!(ξ, ?)
end

=#