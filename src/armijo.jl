

# ----------------------------------- armijo step line search ----------------------------------- #

function armijo(θ, x0, ξ, ζ, Kr, h, Dh, τ;
                resample_dt = 0.001,
                armijo_maxiters = 30,
                show_armijo = false,
                α = 0.4,
                β = 0.7)
                
    γmin = β^armijo_maxiters
    γ = min(γmax(θ,ζ,τ), 1.0)
    φ = ξ
    while γ > γmin
        show_armijo && iiinfo("γ = $(round(γ; digits=6))")
        # φ = P(ξ+γζ)
        φ = armijo_projection(θ, x0, ξ, ζ, γ, Kr, τ; resample_dt)
        g = cost(φ, τ)
        h-g >= -α*γ*Dh ? break : (γ *= β)
    end
    return φ,γ
end

# ----------------------------------- armijo projection ----------------------------------- #

armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ; kw...) = armijo_projection(θ,x0,ξ.x,ξ.u,ζ.x,ζ.u,γ,Kr,τ; kw...)

function armijo_projection(θ::Model{NX,NU},x0,x,u,z,v,γ,Kr,τ; resample_dt=0.001, kw...) where {NX,NU}
    t0,tf = τ; ts = t0:resample_dt:tf
    ubuf = Vector{SVector{NU,Float64}}(undef, length(ts))
    xbuf = Vector{SVector{NX,Float64}}(undef, length(ts))
    i = Ref(1)

    cb = FunctionCallingCallback(funcat = ts, func_start = false) do x1,t,integrator
        (_,x_,u_,z_,v_,γ_,Kr_) = integrator.p

        α = x_(t) + γ_*z_(t)
        μ = u_(t) + γ_*v_(t)
        u1 = μ - Kr_(t)*(x1-α)
        ubuf[i[]] = SVector{NU,Float64}(u1)
        xbuf[i[]] = SVector{NX,Float64}(x1)
        i[] += 1
        # push!(ubuf, SVector{NU,Float64}(u))
        # push!(xbuf, SVector{NX,Float64}(x1))
    end

    ODE(dxdt_armijo, x0, (t0,tf), (θ,x,u,z,v,γ,Kr); callback = cb, dense=false, kw...)
    x = Interpolant(scale(interpolate(xbuf, BSpline(Cubic())), ts))
    u = Interpolant(scale(interpolate(ubuf, BSpline(Cubic())), ts))

    return Trajectory(θ,x,u)
end

function dxdt_armijo(x1, (θ,x,u,z,v,γ,Kr), t)
    α = x(t) + γ*z(t)
    μ = u(t) + γ*v(t)
    Kr = Kr(t)
    u = μ - Kr*(x1-α)
    f(θ,x1,u,t)
end

