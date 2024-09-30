

# ----------------------------------- armijo step line search ----------------------------------- #

function armijo(θ, x0, ξ, ζ, Kr, h, Dh, τ;
                resample_dt = 0.001,
                armijo_maxiters = 25,
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
    t0,tf = τ

    sv = SavedValues(Float64, SVector{NU,Float64})
    cb = SavingCallback(sv) do x1,t,integrator
        (_,x_,u_,z_,v_,γ_,Kr_) = integrator.p
        α = x_(t) + γ_*z_(t)
        μ = u_(t) + γ_*v_(t)
        return SVector{NU,Float64}(μ - Kr_(t)*(x1-α))
    end

    x = ODE(dxdt_armijo, x0, (t0,tf), (θ,x,u,z,v,γ,Kr); callback = cb, kw...)

    u = VecInterpolant(
        MVector{NU, Float64}(undef),
        # [AkimaInterpolation([x[i] for x in sv.saveval], sv.t) for i in 1:NU],
        # [FritschCarlsonMonotonicInterpolation([x[i] for x in sv.saveval], sv.t) for i in 1:NU],
        [CubicSpline([x[i] for x in sv.saveval], sv.t) for i in 1:NU],
    )

    return Trajectory(θ,x,u)
end


struct VecInterpolant{S,T,N,L,ITP}
    buf::MArray{S,T,N,L}
    itps::Vector{ITP}
end

function (x::VecInterpolant)(t)
    for i in eachindex(x.buf)
        x.buf[i] = x.itps[i](t)
    end
    return SVector(x.buf)
end

show(io::IO, x::VecInterpolant) = println(io, make_plot(x, t_plot(x)))
# domain(x::VecInterpolant) = extrema(first(x.itps).knots)
domain(x::VecInterpolant) = domain(first(x.itps))

function armijo_cb!(integrator)
    (_,x_,u_,z_,v_,γ_,Kr_) = integrator.p
    α = x_(t) + γ_*z_(t)
    μ = u_(t) + γ_*v_(t)
    u1 = μ - Kr_(t)*(x1-α)
    push!(ubuf, SVector{NU,Float64}(u1))
end

function dxdt_armijo(x1, (θ,x,u,z,v,γ,Kr), t)
    α = x(t) + γ*z(t)
    μ = u(t) + γ*v(t)
    Kr = Kr(t)
    u = μ - Kr*(x1-α)
    f(θ,x1,u,t)
end

