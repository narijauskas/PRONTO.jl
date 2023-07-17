using PRONTO
using FastClosures
using StaticArrays
using LinearAlgebra
using MatrixEquations



@kwdef struct InvPend <: Model{2,1}
    L::Float64 = 2 # length of pendulum (m)
    g::Float64 = 9.81 # gravity (m/s^2)
end

@dynamics InvPend [
    x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
]

@define_l InvPend begin
    Q = I(2)
    R = I(1)
    1/2*x'*Q*x + 1/2*u'*R*u
    # 1/2*u'*R*u
end

@define_m InvPend begin
    P = [
            88.0233 39.3414;
            39.3414 17.8531;
        ]
    1/2*x'*P*x
end

@define_Q InvPend diagm([10, 1])
@define_R InvPend diagm([1e-3])

resolve_model(InvPend)

PRONTO.runtime_info(θ::InvPend, ξ; verbosity) = nothing
# PRONTO.runtime_info(θ::InvPend, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u; color=PRONTO.manto_colors[1]))
PRONTO.runtime_info(θ::InvPend, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x; color=PRONTO.manto_colors))
# Rreg(x,u,t,θ) = diagm([1e-3])
# Qreg(x,u,t,θ) = diagm([10, 1])

# function stagecost(x,u,t,θ)
#     Ql = I(NX)
#     Rl = I(NU)
#     1/2 * collect(x')*Ql*x + 1/2 * collect(u')*Rl*u
# end

# function termcost(x,u,t,θ)
#     Pl = [
#             88.0233 39.3414;
#             39.3414 17.8531;
#         ]
#     1/2*collect(x')*Pl*x
# end

# PRONTO.generate_model(InvPend, dynamics, stagecost, termcost, Qreg, Rreg)
# ##



##
θ = InvPend()
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]
# smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
μ = t->u0*sin(t)
# α = t->smooth(t, x0, xf, tf)
α = t->xf*t
φ = PRONTO.Trajectory(θ,α,μ);
ξ,data = pronto(θ,x0,φ,τ;tol=1e-4,maxiters=50,verbosity=1);

##
using GLMakie
fig = Figure()
ts = 0:0.001:10

ax = Axis(fig[1,1]; title="optimal trajectory", xlabel="time [s]", ylabel="angle [rad]")
x1 = [data.ξ[end].x(t)[1] for t∈ts]
lines!(ax, ts, x1)

ax = Axis(fig[2,1];xlabel="time [s]", ylabel="angular velocity [rad/s]")
x2 = [data.ξ[end].x(t)[2] for t∈ts]
lines!(ax, ts, x2)

ax = Axis(fig[3,1]; xlabel="time [s]", ylabel="input [Nm]")
u = [data.ξ[end].u(t)[1] for t∈ts]
lines!(ax, ts, u)
display(fig)

save("optimal.png",fig)

##
fig2 = Figure()
ax = Axis(fig2[1,1]; title="descent", xlabel="iteration", ylabel="-Dg", yscale=log10)
lines!(ax, -data.Dh)