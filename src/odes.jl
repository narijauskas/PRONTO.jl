using Memoization

struct ODE{T}
    wrap::FunctionWrapper{T, Tuple{Float64}}
    soln::SciMLBase.AbstractODESolution
    #MAYBE:
    # domain
    # buffer
end

(ode::ODE)(t) = ode.wrap(t)


# try to infer size from the *type* of the initial condition - must use StaticArray
ODE(fn::Function, ic, ts, p; kw...) = ODE(fn::Function, ic, ts, p, Size(ic); kw...)
# constructor basically wraps ODEProblem, should make algorithm available for tuning
function ODE(fn::Function, ic, ts, p, ::Size{S}; kw...) where {S}

    soln = solve(ODEProblem(fn, collect(ic), ts, p),
            Tsit5();
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
Base.eltype(::Type{ODE{T}}) where {T} = T
Base.extrema(ode::ODE) = extrema(ode.soln.t)
domain(ode::ODE; n=240) = extrema(ode)

# stats(ode)
# retcode(ode)
# algorithm(ode)


# T = LinRange(extrema(ode.soln.t)..., 240)
Base.show(io::IO, ode::ODE) = print(io, typeof(ode))
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


preview(ode; kw...) = preview(ode, domain(ode)...; kw...)
preview(fn, t0, tf; kw...) = preview(fn, LinRange(t0, tf, 240); kw...)

function preview(ode, T; height = 30, width = 120, labels = false, kw...)
    x = [ode(t)[i] for t in T, i in 1:length(ode)]
    lineplot(T, x; height, width, labels, kw...)
end

export domain, preview

# this might be type piracy... but it prevents some obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end












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

# this might be type piracy... but it prevents some obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end

# for convenience more than anything
function Base.getindex(ode::ODE, i)
    T = LinRange(extrema(ode.sln.t)..., 1001)
    [ode(t)[ix] for t in T, ix in i]
end

=#