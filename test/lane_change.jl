# to compare against PRONTO for Dummies
import Pkg; Pkg.activate("."); #Pkg.instantiate()
@info "loading dependencies"
using SciMLBase
using OrdinaryDiffEq
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
using MatrixEquations
# using DataInterpolations
@info "loading plots"
using GLMakie; display(lines(rand(10)))
@info "loading PRONTO"
using PRONTO
using PRONTO: jacobian, hessian
@info "ready, running file"
## --------------------------- helper plot trajectories/timeseries --------------------------- ##

include("plot_setup.jl")
function plot_all(t, X, U)

    fig = Figure(resolution = (1800, 1800),)

    ax = Axis(fig[1,1:2]; title = "Lateral Position")
    lines!(ax, t, map(τ->X(τ)[1], t))

    ax = Axis(fig[1,3:4]; title = "Lateral Velocity")
    lines!(ax, t, map(τ->X(τ)[2], t))

    ax = Axis(fig[2,1:2]; title = "Yaw Angle")
    lines!(ax, t, map(τ->X(τ)[3], t))

    ax = Axis(fig[2,3:4]; title = "Angular Velocity")
    lines!(ax, t, map(τ->X(τ)[4], t))

    ax = Axis(fig[3,1:2]; title = "Steering Angle")
    hlines!(ax, deg2rad(30); linestyle=:dash,color=clr[1])
    hlines!(ax, deg2rad(-30); linestyle=:dash,color=clr[1])
    lines!(ax, t, map(τ->X(τ)[5], t); color=clr[1])
    hlines!(ax, deg2rad(6); linestyle=:dash,color=clr[2])
    hlines!(ax, deg2rad(-6); linestyle=:dash,color=clr[2])
    lines!(ax, t, map(τ->X(τ)[6], t); color=clr[2])

    ax = Axis(fig[3,3:4]; title = "Steering Rate")
    hlines!(ax, 1; linestyle=:dash, color=clr_mg)
    hlines!(ax, -1; linestyle=:dash, color=clr_mg)
    lines!(ax, t, map(τ->U(τ)[1], t))
    lines!(ax, t, map(τ->U(τ)[2], t))

    display(fig)
    return fig
end

## --------------------------- problem definition --------------------------- ##

# symbolic states
NX = 6
NU = 2
@variables x[1:NX] u[1:NU]
#NOTE: Symbolics.jl is still actively working on array symbolic variable support. # Things are limited. Things will change.


# dynamics

# model parameters
M = 2041    # [kg]     Vehicle mass
J = 4964    # [kg m^2] Vehicle inertia (yaw)
g = 9.81    # [m/s^2]  Gravity acceleration
Lf = 1.56   # [m]      CG distance, front
Lr = 1.64   # [m]      CG distance, back
μ = 0.8     # []       Coefficient of friction
b = 12      # []       Tire parameter (Pacejka model)
c = 1.285   # []       Tire parameter (Pacejka model)
s = 30      # [m/s]    Vehicle speed

# sideslip angles
αf(x) = x[5] - atan((x[2] + Lf*x[4])/s)
αr(x) = x[6] - atan((x[2] - Lr*x[4])/s)

# tire force
F(α) = μ*g*M*sin(c*atan(b*α))

# model dynamics
function f(x,u)
    # continuous dynamics
    return [
        s*sin(x[3]) + x[2]*cos(x[3]),
        -s*x[4] + ( F(αf(x))*cos(x[5]) + F(αr(x))*cos(x[6]) )/M,
        x[4],
        ( F(αf(x))*cos(x[5])*Lf - F(αr(x))*cos(x[6])*Lr )/J,
        u[1],
        u[2],
    ]
end

# stage cost
Ql = I
Rl = I
l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)


## --------------------------- build model --------------------------- ##

model = (
    t = 0:0.001:10,
    x0 = [-5; zeros(NX-1)],
    x_eq = zeros(NX),
    u_eq = zeros(NU),
    f = f,
    l = l,
    maxiters = 1,
    tol = 1e-3,
    β = 0.7,
    α = 0.4,
);


## --------------------------- regulator parameters --------------------------- ##
ϵ = 0
Qlqr = Timeseries(t->diagm([1,ϵ,1,ϵ,ϵ,ϵ]))
Rlqr = Timeseries(t->0.1*diagm([1,1]))

model = merge(model, (
    Qr = Qlqr,
    Rr = Rlqr,
))


## --------------------------- autodiff --------------------------- ##

model = merge(model, (
    fx = jacobian(x, model.f, x, u),
    fu = jacobian(u, model.f, x, u),
))
model = merge(model, (
    fxx = jacobian(x, model.fx, x, u),
    fxu = jacobian(u, model.fx, x, u),
    fuu = jacobian(u, model.fu, x, u),
))
model = merge(model, (
    lx = jacobian(x, model.l, x, u),
    lu = jacobian(u, model.l, x, u),
))
model = merge(model, (
    lxx = jacobian(x, model.lx, x, u),
    lxu = jacobian(u, model.lx, x, u),
    luu = jacobian(u, model.lu, x, u),
))

## --------------------------- terminal cost --------------------------- ##

xt = model.x_eq
ut = model.u_eq

A = model.fx(xt, ut)
B = model.fu(xt, ut)
Q = model.lxx(xt, ut)
R = model.luu(xt, ut)
S = model.lxu(xt, ut)
Po,_ = arec(A, B, R, Q ,S)
p = x -> 1/2*collect(x)'*Po*collect(x)

model = merge(model, (p=p,))
model = merge(model, (px = jacobian(x, model.p, x),))
model = merge(model, (pxx = jacobian(x, model.px, x),))


## --------------------------- zero initial trajectory --------------------------- ##
u0 = Timeseries(t->zeros(NU))
x0 = Timeseries(t->zeros(NX))
ξ = (x0,u0)


## --------------------------- optimize --------------------------- ##
@time ξ = pronto(ξ, model)
# ProfileView.@profview ξ = pronto(ξ, model)

## --------------------------- plot --------------------------- ##
fig = plot_all(model.t, ξ...)

## --------------------------- testing --------------------------- ##
# P = PRONTO.regulator(ξ..., model)