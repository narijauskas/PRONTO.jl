

armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ; kw...) = armijo_projection(θ,x0,ξ.x,ξ.u,ζ.x,ζ.u,γ,Kr,τ; kw...)

function armijo_projection(θ::Model{NX,NU},x0,x,u,z,v,γ,Kr,τ; dt=0.001, kw...) where {NX,NU}
    ubuf = Vector{SVector{NU,Float64}}()
    t0,tf = τ; ts = t0:dt:tf

    cb = FunctionCallingCallback(funcat = ts, func_start = false) do x1,t,integrator
        (θ,x,u,z,v,γ,Kr) = integrator.p

        α = x(t) + γ*z(t)
        μ = u(t) + γ*v(t)
        Kr = Kr(t)
        u = μ - Kr*(x1-α)

        push!(ubuf, SVector{NU,Float64}(u))
    end

    x = ODE(dxdt_armijo, x0, (t0,tf), (θ,x,u,z,v,γ,Kr); callback = cb, saveat = ts, kw...)
    u = Interpolant(scale(interpolate(ubuf, BSpline(Linear())), ts))

    return Trajectory(θ,x,u)
end

function dxdt_armijo(x1, (θ,x,u,z,v,γ,Kr), t)
    α = x(t) + γ*z(t)
    μ = u(t) + γ*v(t)
    Kr = Kr(t)
    u = μ - Kr*(x1-α)
    f(x1,u,t,θ)
end
