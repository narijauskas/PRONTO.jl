
#=
to allow more interpolant types in the future:
    import SciMLBase: AbstractDiffEqInterpolation as AbstractInterpolation
    import SciMLBase: LinearInterpolation, ConstantInterpolation
or possibly:
    abstract type AbstractInterpolation end

for now: 
=#
import SciMLBase: LinearInterpolation

# intermediate type to prevent type piracy and unwanted behaviors
struct Interpolation #<:AbstractArray?
    id::LinearInterpolation
end

Interpolation(t,u) = Interpolation(LinearInterpolation(t,u))

(itp::Interpolation)(tvals) = SciMLBase.interpolation(tvals,itp.id,nothing,Val{0},nothing,:left)
(itp::Interpolation)(val,tvals) = SciMLBase.interpolation!(val,tvals,itp.id,nothing,Val{0},nothing,:left)

Base.setindex!(itp::Interpolation, X, inds...) = setindex!(itp.id.u, X, inds...)



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
    t # t::Vector{Float64} ??
    
    Trajectory(X,U,t) = new(
        Interpolation(t, map(τ->X(τ),t)),
        Interpolation(t, map(τ->U(τ),t)),
        t
    )
end

# ξ.x(t), ξ.u(t) provided by Interpolation

# evaluate X(t) for each t in ξ.t, store to ξ.x at t
function update_x!(ξ, X)
    # map!(t->X(t), ξ.x, ξ.t)
    for (i,t) in enumerate(ξ.t)
        setindex!(ξ.x, X(t), i)
    end
end

function update_u!(ξ, U)
    for (i,t) in enumerate(ξ.t)
        setindex!(ξ.u, U(t), i)
    end
end


















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

