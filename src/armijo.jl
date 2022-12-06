


cost(ξ,τ) = cost(ξ.θ,ξ.x,ξ.u,τ)

function cost(θ,x,u,τ)
    t0,tf = τ; h0 = SVector{1,Float64}(0)
    hf = p(x(tf),u(tf),tf,θ)
    h = hf + solve(ODEProblem(dh_dt, h0, (t0,tf), (θ,x,u)), Tsit5(); reltol=1e-7)(tf)
    return h[1]
end

dh_dt(h, (θ,x,u), t) = l(x(t), u(t), t, θ)





armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ; kw...) = armijo_projection(θ,x0,ξ.x,ξ.u,ζ.x,ζ.u,γ,Kr,τ; kw...)
function armijo_projection(θ::Model{NX,NU},x0,x,u,z,v,γ,Kr,τ; dt=0.001) where {NX,NU}
    ubuf = Vector{SVector{NU,Float64}}()
    t0,tf = τ; ts = t0:dt:tf

    cb = FunctionCallingCallback(funcat = ts, func_start = false) do x1,t,integrator
        (θ,x,u,z,v,γ,Kr) = integrator.p

        α = x(t) + γ*z(t)
        μ = u(t) + γ*v(t)
        Kr = Kr(α,μ,t)
        u = μ - Kr*(x1-α)

        push!(ubuf, SVector{NU,Float64}(u))
    end

    x = ODE(dxdt_armijo, x0, (t0,tf), (θ,x,u,z,v,γ,Kr); callback = cb, saveat = ts)
    u = Interpolant(scale(interpolate(ubuf, BSpline(Linear())), ts))

    return Trajectory(θ,x,u)
end


function dxdt_armijo(x1, (θ,x,u,z,v,γ,Kr), t)
    α = x(t) + γ*z(t)
    μ = u(t) + γ*v(t)
    Kr = Kr(α,μ,t)
    u = μ - Kr*(x1-α)
    f(x1,u,t,θ)
end

# u_armijo(ξ,ζ,γ,Kr,t)
# function u_armijo(x,u,z,v,γ,Kr,t)
#     α = x(t) + γ*z(t)
#     μ = u(t) + γ*v(t)
#     Kr = Kr(α,μ,t)
#     u = μ - Kr*(x-α)
# end