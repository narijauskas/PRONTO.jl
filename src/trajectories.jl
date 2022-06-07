
#=
to allow more interpolant types in the future:
    import SciMLBase: AbstractDiffEqInterpolation as AbstractInterpolation
    import SciMLBase: LinearInterpolation, ConstantInterpolation
or possibly:
    abstract type AbstractInterpolation end

for now: 
=#
import SciMLBase: LinearInterpolation

# intermediate type to add custom functionality, and prevent type piracy and unwanted behaviors
# struct Interpolant{T}
#     itp::LinearInterpolation
#     t::Ref

#     function Interpolant(f,t)
#         itp = LinearInterpolation(t, map(τ->f(τ),t))
#         t = Ref(itp.t)
#         new(itp, t)
#     end
# end


struct Interpolant#{T}
    itp::LinearInterpolation
    t
    Interpolant(f,t) = new(LinearInterpolation(t, map(τ->f(τ),t)), t)
end


#X(t)
(X::Interpolant)(tvals) = SciMLBase.interpolation(tvals,X.itp,nothing,Val{0},nothing,:left)
(X::Interpolant)(val,tvals) = SciMLBase.interpolation!(val,tvals,X.itp,nothing,Val{0},nothing,:left)
# (X::Interpolant{T})(tvals) where {T} = SciMLBase.interpolation(tvals,X.itp,nothing,Val{0},nothing,:left)::T
# (X::Interpolant{T})(val,tvals) where {T} = SciMLBase.interpolation!(val,tvals,X.itp,nothing,Val{0},nothing,:left)::T

# update values of X by evaluating f(t) for each t
function update!(f, X::Interpolant)
    for (i,t) in enumerate(itp.t)
        setindex!(X, f(t), i)
    end
end

Base.setindex!(X::Interpolant, val, inds...) = setindex!(X.itp.u, val, inds...)



#MAYBE: by making it an abstractarray subtype, we might get all sorts of functionality for free






























# allows for x(t)

# defines the Trajectory type
# used to represent objects of the form ξ = (x(t),u(t))
# x and u are sampled to a fixed timestep
# based on a linear interpolant object (maybe more options in future)

# x and u are mutable, while the trajectory is not

# x and u are interpolants on StaticArrays? Arrays?
# have constant, known return types XT,UT?

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

