
# ODEBuffer{N} = MArray{N, Float64}
# ODEBuffer{N}() where {N} = MArray{N, Float64}(undef)

struct ODE{T}
    
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    sln::SciMLBase.AbstractODESolution
end

(ode::ODE)(t) = ode.fxn(t)

# trajectories are DAEs, pass the mass matrix via the kwarg dae=()
dae(M)::Matrix{Float64} = mass_matrix(M)
mass_matrix(M) = cat(diagm(ones(nx(M))), diagm(zeros(nu(M))); dims=(1,2))

# constructor basically wraps ODEProblem
function ODE(fn, x0, ts, ode_pm, buf::T; dae=nothing, ode_kw...) where {T}
    ode_fn = isnothing(dae) ? ODEFunction(fn) : ODEFunction(fn; mass_matrix=dae)
    sln = solve(ODEProblem(ode_fn,x0,ts,ode_pm; ode_kw...); reltol=1e-7, saveat=0.001)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->copy(sln(buf,t)))
    ODE{T}(fxn,buf,sln)
end

Base.size(ode::ODE) = size(ode.buf)

function _preview(ode::ODE)
    T = LinRange(extrema(ode.sln.t)..., 240)
    x = [ode(t)[i] for t in T, i in 1:length(ode.buf)]
    lineplot(T, x; height=20, width=80, labels=false)
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

function Base.show(io::IO, ode::ODE)
    compact = get(io, :compact, false)
    print(io, typeof(ode))
    if !compact
        println(io)
        print(io, preview(ode))
    end
    return nothing
end
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
