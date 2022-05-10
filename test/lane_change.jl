# to compare against PRONTO for Dummies
import Pkg; Pkg.activate(".")
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
using MatrixEquations
using DataInterpolations

using GLMakie; display(lines(rand(10)))
using PRONTO
using PRONTO: jacobian, hessian

## --------------------------- helper plot trajectories/timeseries --------------------------- ##

function plot_timeseries!(ax, t, X, n; kw...)
    for i in 1:n
        lines!(ax, t, map(τ->X(τ)[i], t); kw...)
    end
    return ax
end

function plot_timeseries(t, X, n; kw...)
    fig = Figure(; kw...)
    ax = Axis(fig[1,1]; kw...)

    plot_timeseries!(ax, t, X, n; kw...)
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
    maxiters = 100,
    tol = 1e-3,
    β = 0.7,
    α = 0.4,
);



## --------------------------- regulator parameters --------------------------- ##

# Qr = Timeseries(t->diagm([10,1]), model.t) # needs to capture X(t)
# Rr = Timeseries(t->1e-3, model.t) # needs to capture X(t)

ϵ = 0
Qlqr = Timeseries(t->diagm([1,ϵ,1,ϵ,ϵ,ϵ]), model.t)
Rlqr = Timeseries(t->0.1*diagm([1,1]), model.t)

model = merge(model, (
    Qr = Qlqr,
    Rr = Rlqr,
))

## --------------------------- autodiff & terminal cost --------------------------- ##
include("build_model.jl");


## --------------------------- zero initial trajectory --------------------------- ##
u0 = Timeseries(t->zeros(NU), model.t)
x0 = Timeseries(t->zeros(NX), model.t)
ξ = (x0,u0)


## --------------------------- or, solve zero input dynamics --------------------------- ##
# function dynamics!(dx, x, u, t)
#     dx .= f(x, u(t))
# end

# T = last(model.t)
# x0 = solve(ODEProblem(dynamics!, model.x0, (0, T), u0))
# x0 = Timeseries(t->x0(t), model.t)
# ξ = (x0,u0)

## --------------------------- optimize --------------------------- ##
@time ξ = pronto(ξ, model)


## --------------------------- plot --------------------------- ##
plot_timeseries(model.t, ξ[1], 6)

