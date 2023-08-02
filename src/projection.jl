
#MAYBE: move to trajectories.jl?
struct Trajectory{M,X,U}
    θ::M
    x::X
    u::U
end

(ξ::Trajectory)(t) = (x=ξ.x(t),u=ξ.u(t))
# Base.show(io::IO, ξ::Trajectory) = print(io, "Trajectory")
nx(ξ::Trajectory) = nx(ξ.θ)
nu(ξ::Trajectory) = nu(ξ.θ)
extrema(ξ::Trajectory) = extrema(ξ.x)
domain(ξ::Trajectory) = domain(ξ.x)

show(x::DataInterpolations.CubicSpline{FT,T}) where {FT,T} = println(io, make_plot(x, t_plot(x)))
show(x::DataInterpolations.AbstractInterpolation{FT,T}) where {FT,T} = println(io, make_plot(x, t_plot(x)))
domain(x::DataInterpolations.AbstractInterpolation) = extrema(x.t)

function show(io::IO, ξ::Trajectory)
    println(io, make_plot(ξ.x, t_plot(ξ.x)))
    println(io, make_plot(ξ.u, t_plot(ξ.u)))
end

function projection(θ::Model, x0, η, τ; kw...)
    Kr = regulator(θ, η, τ)
    projection(θ, x0, η, Kr, τ; kw...)
end

projection(θ::Model, x0, η, Kr, τ; kw...) = projection(θ, x0, η.x, η.u, Kr, τ; kw...)


function projection(θ::Model{NX,NU}, x0, α, μ, Kr, (t0,tf); dt=0.001) where {NX,NU}
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

    
    # sv = SavedValues(Float64, SVector{NU,Float64})
    # cb = SavingCallback(sv) do x,t,integrator
    #     (_,α,μ,Kr) = integrator.p

    #     α = α(t)
    #     μ = μ(t)
    #     Kr = Kr(α,μ,t)
    #     return SVector{NU,Float64}(μ - Kr*(x-α))
    # end

    # x = ODE(dxdt, x0, (t0,tf), (θ,α,μ,Kr); callback = cb)
    # u = DataInterpolations.CubicSpline(sv.saveval, sv.t)
    # u = DataInterpolations.AkimaInterpolation(first.(sv.saveval), sv.t)

    # u = VecInterpolant(
    #     MVector{NU, Float64}(undef),
    #     [CubicSpline([x[i] for x in sv.saveval], sv.t) for i in 1:NU],
    #     # [interpolate(sv.t, [x[i] for x in sv.saveval], SteffenMonotonicInterpolation()) for i in 1:NU]
    # )

    ODE(dxdt, x0, (t0,tf), (θ,α,μ,Kr); dense=false, callback = cb)
    x = Interpolant(scale(interpolate(xbuf, BSpline(Cubic())), ts))
    u = Interpolant(scale(interpolate(ubuf, BSpline(Cubic())), ts))


    return Trajectory(θ,x,u)
end


function dxdt(x, (θ,α,μ,Kr), t)
    α = α(t)
    μ = μ(t)
    Kr = Kr(α,μ,t)
    u = μ - Kr*(x-α)
    f(θ,x,u,t)
end
