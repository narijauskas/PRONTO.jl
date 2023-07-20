
γmax(θ::Model,ζ,τ) = 1.0

# ----------------------------------- armijo step line search ----------------------------------- #

function armijo(θ, x0, ξ, ζ, Kr, h, Dh, τ; verbosity, armijo_maxiters)
    # h = cost(ξ, τ)
    α=0.4; β=0.7
    γmin = β^armijo_maxiters
    γ = min(γmax(θ,ζ,τ), 1.0)
    # γ = 1.0
    φ = ξ
    while γ > γmin
        iiinfo("armijo γ = $(round(γ; digits=6))"; verbosity)
        φ = armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)
        g = cost(φ, τ)
        h-g >= -α*γ*Dh ? break : (γ *= β)
    end
    return φ,γ
end

# ----------------------------------- armijo projection ----------------------------------- #

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
    f(θ,x1,u,t)
end
