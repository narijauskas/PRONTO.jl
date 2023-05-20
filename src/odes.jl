# plot handling

# ----------------------------------- interpolation functor ----------------------------------- #

# used to 
struct Interpolant{T}
    itp::T
end
(x::Interpolant{T})(t) where {T} = x.itp(t)
export Interpolant

show(io::IO, ::Type{Interpolant{T}}) where {T} = print(io, "Interpolant{$(eltype(T)),...}")
show(io::IO, u::Interpolant) = print(io, preview(u))
# Base.length(::Interpolant{T}) where {T} = length(eltype(T))
eltype(::Interpolant{T}) where {T} = eltype(T)
extrema(u::Interpolant) = extrema(first(u.itp.ranges))
eachindex(::Interpolant{T}) where {T} = OneTo(length(eltype(T)))


# ----------------------------------- ODE solution wrapper ----------------------------------- #

struct ODE{T}
    wrap::FunctionWrapper{T, Tuple{Float64}}
    soln::SciMLBase.AbstractODESolution
end

(ode::ODE)(t) = ode.wrap(t)


# try to infer size from the *type* of the initial condition - must use StaticArray
ODE(fn::Function, ic, ts, p; kw...) = ODE(fn::Function, ic, ts, p, Size(ic); kw...)
# constructor basically wraps ODEProblem, should make algorithm available for tuning
function ODE(fn::Function, ic, ts, p, ::Size{S}; alg=Tsit5(), kw...) where {S}

    soln = solve(ODEProblem(fn, ic, ts, p),
            alg;
            reltol=1e-7, kw...)

    T = SArray{Tuple{S...}, Float64, length(S), prod(S)}

    wrap = FunctionWrapper{T, Tuple{Float64}}() do t
            out = MArray{Tuple{S...}, Float64, length(S), prod(S)}(undef)
            soln(out,t)
            return SArray{Tuple{S...}, Float64, length(S), prod(S)}(out)
    end

    ODE{T}(wrap,soln)
end


# might not be needed
StaticArrays.Size(::ODE{T}) where {T} = Size(T)
StaticArrays.Size(::Type{ODE{T}}) where {T} = Size(T)
Base.size(::ODE{T}) where {T} = size(T)
Base.length(::ODE{T}) where {T} = length(T)

eltype(::Type{ODE{T}}) where {T} = T
extrema(ode::ODE) = extrema(ode.soln.t)
eachindex(::ODE{T}) where {T} = OneTo(length(T))
show(io::IO, ode::ODE) = println(io, preview(ode))

# stats(ode)
# retcode(ode)
# algorithm(ode)


# T = LinRange(extrema(ode.soln.t)..., 240)
# Base.show(io::IO, ode::ODE) = print(io, typeof(ode))

#TODO: more info
# function Base.show(io::IO, ode::ODE)
#     compact = get(io, :compact, false)
#     if compact
#         print(io, typeof(ode))
#     else
#         println(io)
#         print(io, preview(ode))
#         println(io)
#     end
#     return nothing
#     # print(io, typeof(ode))
#     # if !compact
#     #     println(io)
#     #     print(io, preview(ode))
#     # end
#     # return nothing
# end


# preview(ode; kw...) = preview(ode, domain(ode)...; kw...)
# preview(fn, t0, tf; kw...) = preview(fn, LinRange(t0, tf, 240); kw...)

# function preview(ode, T; height = 30, width = 120, labels = false, kw...)
#     x = [ode(t)[i] for t in T, i in 1:length(ode)]
#     lineplot(T, x; height, width, labels, kw...)
# end

export domain, preview

# this is type piracy... but it prevents some obscenely long error messages
function show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end

# ----------------------------------- ODE display ----------------------------------- #
# I find it much more intuitive to show a trace of an ODE solution

PLOT_HEIGHT::Int = 12
PLOT_WIDTH::Int = 120
t_plot(t0,tf) = LinRange(t0,tf,4*PLOT_WIDTH)
t_plot(x) = t_plot(extrema(x)...)

function set_plot_scale(height, width)
    global PLOT_HEIGHT = convert(Int, height)
    global PLOT_WIDTH = convert(Int, width)
end


preview(x; kw...) = preview(x, t_plot(x), eachindex(x); kw...)

function preview(x, ts, is; kw...)
    lineplot(ts, [x(t)[i] for t∈ts, i∈is];
                height = PLOT_HEIGHT,
                width = PLOT_WIDTH,
                labels = false,
                kw...)
end





#FUTURE: Makie support
# using MakieCore
# function MakieCore.convert_arguments(P::MakieCore.PointBased, x::ODE, i)
#     ts = LinRange(extrema(x)...,1001)
#     # is = eachindex(x)
#     xs = [x(t)[i] for t∈ts]
#     (MakieCore.convert_arguments(P, collect(ts), xs),
#     MakieCore.convert_arguments(P, collect(ts), xs))
# end







#=







# trajectories are DAEs, pass the mass matrix via the kwarg dae=()
dae(M)::Matrix{Float64} = mass_matrix(M)
mass_matrix(M) = cat(diagm(ones(nx(M))), diagm(zeros(nu(M))); dims=(1,2))



# fnw = FunctionWrapper{T, Tuple{Real}}() do t
#     out = MArray{...}(undef)
#     sln(out,t)
#     return SArray(out)
# end


@inline set_static(A) = A
@inline set_static(A::MArray) = SVector(A)


function _preview(ode::ODE)
    T = LinRange(extrema(ode.soln.t)..., 240)
    x = [ode(t)[i] for t in T, i in 1:length(ode)]
    lineplot(T, x; height=30, width=120, labels=false)
end

function preview(ode::ODE)
    println(stdout)
    println(stdout,_preview(ode))
    return nothing
end

function plot_trajectory(M, ode::ODE)
    T = LinRange(extrema(ode.sln.t)..., 240)
    x = [ode(t)[i] for t in T, i in 1:nx(M)]
    u = [ode(t)[i] for t in T, i in nx(M)+1:nx(M)+nu(M)]
    
    println(stdout)
    println(stdout, lineplot(T, x; height=20, width=80))
    println(stdout, lineplot(T, u; height=20, width=80))
    return nothing
end

# function Base.show(io::IO, ode::ODE)
#     compact = get(io, :compact, false)
#     print(io, typeof(ode))
#     if !compact
#         println(io)
#         print(io, preview(ode))
#     end
#     return nothing
# end
#FUTURE: show size, length, time span, solver method?

# this is type piracy... but it prevents some obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end

# for convenience more than anything
function Base.getindex(ode::ODE, i)
    T = LinRange(extrema(ode.sln.t)..., 1001)
    [ode(t)[ix] for t in T, ix in i]
end

=#