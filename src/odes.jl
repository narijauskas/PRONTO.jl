# plot handling

# ----------------------------------- interpolation functor ----------------------------------- #

struct Interpolant{T}
    itp::T
end
(x::Interpolant{T})(t) where {T} = x.itp(t)
export Interpolant

show(io::IO, ::Type{Interpolant{T}}) where {T} = print(io, "Interpolant{$(eltype(T)),...}")
show(io::IO, u::Interpolant) = println(io, make_plot(u, t_plot(u)))
# Base.length(::Interpolant{T}) where {T} = length(eltype(T))
eltype(::Interpolant{T}) where {T} = eltype(T)
domain(u::Interpolant) = extrema(first(u.itp.ranges))
eachindex(::Interpolant{T}) where {T} = OneTo(length(eltype(T)))


# ----------------------------------- ODE solution wrappers ----------------------------------- #
abstract type AbstractODE end

struct BufferedODE{S,T,N,L,SType} <: AbstractODE
    buf::MArray{S,T,N,L}
    soln::SType
end

function (ode::BufferedODE{S,T,N,L})(t) where {S,T,N,L}
    fill!(ode.buf, 0)
    ode.soln(ode.buf, t)
    return SArray{S,T,N,L}(ode.buf)
end


# ODE(fn::Function, ic, ts, p; kw...) = ODE(fn::Function, ic, ts, p, Size(ic); kw...)

# function ODE(fn::Function, ic, ts, p, ::Size{S}; alg=Tsit5(), kw...) where {S}
#     buf = MArray{Tuple{S...}, Float64, length(S), prod(S)}(undef)
#     # fill!(buf, 0)
#     soln = solve(ODEProblem(fn, ic, ts, p),
#             alg;
#             abstol=1e-8,
#             kw...)
#     return BufferedODE(buf, soln)
# end

show(io::IO, x::BufferedODE) = println(io, make_plot(x, t_plot(x)))
domain(x::BufferedODE) = extrema(x.soln.interp.ts)


struct WrappedODE{T} <: AbstractODE
    wrap::FunctionWrapper{T, Tuple{Float64}}
    soln::SciMLBase.AbstractODESolution
end

(ode::WrappedODE)(t) = ode.wrap(t)
# (ode::ODE{T})(t) where {T} = T(ode.soln(t))

# try to infer size from the *type* of the initial condition - must use StaticArray
WrappedODE(fn::Function, ic, ts, p; kw...) = WrappedODE(fn::Function, ic, ts, p, Size(ic); kw...)
# constructor basically wraps ODEProblem, should make algorithm available for tuning
function WrappedODE(fn::Function, ic, ts, p, ::Size{S}; alg=Tsit5(), kw...) where {S}

    soln! = solve(ODEProblem(fn, ic, ts, p),
            alg;
            # reltol=1e-8,
            abstol=1e-8,
            kw...)

    T = SArray{Tuple{S...}, Float64, length(S), prod(S)}

    wrap = FunctionWrapper{T, Tuple{Float64}}() do t
            out = MArray{Tuple{S...}, Float64, length(S), prod(S)}(undef)
            soln!(out,t)
            return SArray{Tuple{S...}, Float64, length(S), prod(S)}(out)
    end

    WrappedODE{T}(wrap,soln!)
end


# might not be needed
# StaticArrays.Size(::ODE{T}) where {T} = Size(T)
# StaticArrays.Size(::Type{ODE{T}}) where {T} = Size(T)
# Base.size(::ODE{T}) where {T} = size(T)
# Base.length(::ODE{T}) where {T} = length(T)

# eltype(::Type{ODE{T}}) where {T} = T
# eachindex(::ODE{T}) where {T} = OneTo(length(T))
domain(x::WrappedODE) = extrema(x.soln.t)
show(io::IO, x::WrappedODE) = println(io, make_plot(x, t_plot(x)))

# domain(ode) = extrema(ode.t)
# stats(ode)
# retcode(ode)
# algorithm(ode)


# this is type piracy... but it prevents some obscenely long error messages
function show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end


export SlimODE
struct SlimODE{S,T,N,L,RC,IType} <: AbstractODE
    buf::MArray{S,T,N,L}
    retcode::RC
    interp::IType
end

function (ode::SlimODE{S,T,N,L})(t) where {S,T,N,L}
    ode.interp(ode.buf, t, nothing, Val{0}, nothing, :left)
    return SArray{S,T,N,L}(ode.buf)
end

ODE(fn::Function, ic, ts, p; kw...) = ODE(fn::Function, ic, ts, p, Size(ic); kw...)

function ODE(fn::Function, ic, ts, p, ::Size{S}; alg=Tsit5(), kw...) where {S}
    buf = MArray{Tuple{S...}, Float64, length(S), prod(S)}(undef)
    fill!(buf, 0)
    soln = solve(ODEProblem(fn, ic, ts, p),
            alg;
            abstol=1e-8,
            kw...)
    return SlimODE(buf, soln.retcode, soln.interp)
end

show(io::IO, x::SlimODE) = println(io, make_plot(x, t_plot(x)))
domain(x::SlimODE) = extrema(x.interp.ts)

# ----------------------------------- curve display ----------------------------------- #

PLOT_HEIGHT::Int = 10
PLOT_WIDTH::Int = 100
PLOT_COLORS = [
    crayon"#FFC12E",
    crayon"#FF7F10",
    crayon"#FF3C38",
    crayon"#FF006E",
    crayon"#2AD599",
    crayon"#007DC6",
]
PLOT_KW = (
    height = PLOT_HEIGHT, 
    width = PLOT_WIDTH,
    color = repeat(PLOT_COLORS, 3),
)
t_plot(t0,tf) = LinRange(t0,tf,4*PLOT_WIDTH)
t_plot(x) = t_plot(domain(x)...)


# for convenience more than anything
# function Base.getindex(ode::ODE, i)
#     [ode(t)[ix] for t in t_plot(ode), ix in i]
# end


function set_plot_scale(height, width)
    global PLOT_HEIGHT = convert(Int, height)
    global PLOT_WIDTH = convert(Int, width)
end


plot_preview(θ, ξ) = println(make_plot(t->preview(θ, ξ(t)), t_plot(ξ)))

# make_plot(t->vec(Kr(t)), t_plot(ξ))
function make_plot(f, ts)
    lineplot(ts, reduce(hcat, collect.(f(t) for t∈ts))';
                height = PLOT_HEIGHT,
                width = PLOT_WIDTH,
                color = repeat(PLOT_COLORS,3),
                labels = false)
end

# before julia 1.9:
# reduce(hcat, collect.(ξ.x(t) for t∈ts))'
# after julia 1.9:
# stack(ξ.x(t) for t∈ts)'

