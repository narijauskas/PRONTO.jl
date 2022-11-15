
# ----------------------------------- ode solution handling ----------------------------------- #

#TODO: abstract type AbstractBuffer{T}

# maps t->x::T
struct Buffer{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    fn!::Function
end

(buf::Buffer)(t) = buf.fxn(t)

function Buffer(fn!, N::Vararg{Int})
    @assert length(N) >= 1
    T = MArray{Tuple{N...}, Float64, length(N), prod(N)}
    buf = T(undef)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->(fn!(buf, t); return copy(buf)))
    Buffer(fxn,buf,fn!)
end


# maps t->x::T
struct Solution{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    sln::SciMLBase.AbstractODESolution
end

(sln::Solution)(t) = sln.fxn(t)

# # T = BufferType(S...)
# function Solution(prob, N::Vararg{Int})
#     @assert length(N) >= 1
#     sln = solve(prob)
#     T = MArray{Tuple{N...}, Float64, length(N), prod(N)}
#     buf = T(undef)
#     fxn = FunctionWrapper{T, Tuple{Float64}}(t->(sln(buf, t); return copy(buf)))
#     Solution(fxn,buf,sln)
# end


Solution(args...) = @error "please specify a buffer size, eg. Solution{@buffer(NX,NX)...}(args...)"

function Solution{T}(ode_fn,x0,ts,ode_pm; ode_kw...) where {T<:MArray}
    fn = ODEFunction(ode_fn)
    sln = solve(ODEProblem(fn,x0,ts,ode_pm; ode_kw...))
    buf = T(undef)
    # buf = MArray{@buffer(nx(M))...}(undef)
    function xfxn(t)
        sln(buf,t)
        copy(buf)
    end
    fxn = FunctionWrapper{T, Tuple{Float64}}(xfxn)

    Solution{T}(fxn,buf,sln)
end


Base.size(sln::Solution) = size(sln.buf)



function preview(ξ::Solution)
    T = LinRange(extrema(ξ.sln.t)..., 1001)
    x = [ξ(t)[i] for t in T, i in 1:length(ξ.buf)]
    lineplot(T,x; height=20, width=80)
end




macro buffer(N...)
    N = collect(esc.(N))
    :(Tuple{$(N...)}, Float64, length($N), prod([$(N...)]))
end

# maps t->ξ=(x,u)::(TX,TU)
struct Trajectory{TX,TU}
    x::FunctionWrapper{TX, Tuple{Float64}}
    u::FunctionWrapper{TU, Tuple{Float64}}
    xbuf::TX
    ubuf::TU
    sln::SciMLBase.AbstractODESolution
end

(ξ::Trajectory)(t) = vcat(ξ.x(t), ξ.u(t))


#eg. Trajectory(M, ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr))
function Trajectory(M::Model, ode_fn, ξ0, T, ode_pm; ode_kw...)
    fn = ODEFunction(ode_fn; mass_matrix = PRONTO.mass_matrix(M))
    sln = solve(ODEProblem(fn, ξ0, T, ode_pm; ode_kw...))

    xbuf = MArray{@buffer(nx(M))...}(undef)
    function xfxn(t)
        sln(xbuf,t; idxs=1:nx(M))
        # return SArray(xbuf)
        copy(xbuf)
    end
    x = FunctionWrapper{MArray{@buffer(nx(M))...}, Tuple{Float64}}(xfxn)

    ubuf = MArray{@buffer(nu(M))...}(undef)
    function ufxn(t)
        sln(ubuf,t; idxs=(nx(M)+1):(nx(M)+nu(M)))
        # return SArray(ubuf)
        copy(ubuf)
    end
    u = FunctionWrapper{MArray{@buffer(nu(M))...}, Tuple{Float64}}(ufxn)

    Trajectory(x,u,xbuf,ubuf,sln)
end


function preview(ξ::Trajectory)
    T = LinRange(extrema(ξ.sln.t)..., 1001)
    x = [ξ.x(t)[i] for t in T, i in CartesianIndices(size(ξ.xbuf))]
    lineplot(T,x; height=20, width=80)
end

Base.show(io::IO, buf::Buffer) = show(io,typeof(buf))
Base.show(io::IO, sln::Solution) = show(io,typeof(sln))
#FUTURE: show size, length, time span, solver method?
function Base.show(io::IO, ξ::Trajectory)
    compact = get(io, :compact, false)
    print(io, typeof(ξ))
    if !compact
        println()
        print(io,preview(ξ))
    end
end
#FUTURE: show size, length, time span, solver method?

# this might be type piracy... but prevents some obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end

