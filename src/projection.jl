
# export projection


struct Trajectory{M,X,U}
    θ::M
    x::X
    u::U
end

(ξ::Trajectory)(t) = ξ.x(t),ξ.u(t)
# Base.show(io::IO, ξ::Trajectory) = print(io, "Trajectory")
nx(ξ::Trajectory) = nx(ξ.θ)
nu(ξ::Trajectory) = nu(ξ.θ)
extrema(ξ::Trajectory) = extrema(ξ.x)

function show(io::IO, ξ::Trajectory)
    println(io, preview(ξ.x))
    println(io, preview(ξ.u))
end

function projection(θ::Model, x0, η, τ; kw...)
    Kr = regulator(θ, η, τ; verbosity)
    projection(θ, x0, η, Kr, τ; kw...)
end

projection(θ::Model, x0, η, Kr, τ; kw...) = projection(θ, x0, η.x, η.u, Kr, τ; kw...)

#TODO: a function that returns buf,cb

function projection(θ::Model{NX,NU}, x0, α, μ, Kr, (t0,tf); verbosity=1, dt=0.0001) where {NX,NU}
    iinfo("projection"; verbosity)
    xbuf = Vector{SVector{NX,Float64}}()
    ubuf = Vector{SVector{NU,Float64}}()
    ts = t0:dt:tf

    cb = FunctionCallingCallback(funcat = ts, func_start = false) do x,t,integrator
        (_,α,μ,Kr) = integrator.p

        α = α(t)
        μ = μ(t)
        Kr = Kr(α,μ,t)
        u = μ - Kr*(x-α)
        push!(xbuf, SVector{NX,Float64}(x))
        push!(ubuf, SVector{NU,Float64}(u))
    end  

    x = ODE(dxdt, x0, (t0,tf), (θ,α,μ,Kr); callback = cb, saveat = ts)
    # x = Interpolant(scale(interpolate(xbuf, BSpline(Linear())), ts))
    u = Interpolant(scale(interpolate(ubuf, BSpline(Linear())), ts))

    return Trajectory(θ,x,u)
end



function dxdt(x, (θ,α,μ,Kr), t)
    α = α(t)
    μ = μ(t)
    Kr = Kr(α,μ,t)
    u = μ - Kr*(x-α)
    f(θ,x,u,t)
end




