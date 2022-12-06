
export zero_input, open_loop, projection


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





function zero_input(θ::Model{NX,NU}, x0, τ) where {NX,NU}
    μ = t -> zeros(SVector{NU})
    open_loop(θ, x0, μ, τ)
end


function open_loop(θ::Model{NX,NU}, x0, μ, τ) where {NX,NU}
    α = t -> zeros(SVector{NX})
    Kr = (α,μ,t) -> zeros(SMatrix{NU,NX})
    projection(θ, x0, α, μ, Kr, τ)
end


projection(θ::Model, x0, φ, Kr, τ) = projection(θ, x0, φ.x, φ.u, Kr, τ)


#TODO: a function that returns buf,cb

function projection(θ::Model{NX,NU}, x0, α, μ, Kr, (t0,tf); dt=0.001) where {NX,NU}
    # xbuf = Vector{SVector{NX,Float64}}()
    ubuf = Vector{SVector{NU,Float64}}()
    ts = t0:dt:tf

    cb = FunctionCallingCallback(funcat = ts, func_start = false) do x,t,integrator
        (_,α,μ,Kr) = integrator.p

        α = α(t)
        μ = μ(t)
        Kr = Kr(α,μ,t)
        u = μ - Kr*(x-α)
        # push!(xbuf, SVector{NX,Float64}(x))
        push!(ubuf, SVector{NU,Float64}(u))
    end

    x = ODE(dxdt, x0, (t0,tf), (θ,α,μ,Kr); callback = cb, saveat = ts)
    u = Interpolant(scale(interpolate(ubuf, BSpline(Linear())), ts))

    return Trajectory(θ,x,u)
end



function dxdt(x, (θ,α,μ,Kr), t)
    α = α(t)
    μ = μ(t)
    Kr = Kr(α,μ,t)
    u = μ - Kr*(x-α)
    f(x,u,t,θ)
end




