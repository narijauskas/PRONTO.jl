# to compare against PRONTO for Dummies
import Pkg; Pkg.activate(".")
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
# using DataInterpolations
# using DifferentialEquations
using MatrixEquations

using GLMakie; display(lines(rand(10)))
using PRONTO
using PRONTO: jacobian, hessian


## --------------------------- problem definition --------------------------- ##

# symbolic states
nx = 2; nu = 1
@variables x[1:nx] u[1:nu]
#NOTE: Symbolics.jl is still actively working on array symbolic variable support. # Things are limited. Things will change.


# dynamics
g = 9.81
L = 2
f = (x,u) -> [x[2];   g/L*sin(x[1])-u[1]*cos(x[1])/L]


# stage cost
Ql = I
Rl = I
l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)


model = (
    t = 0:0.001:10,
    x0 = [2π/3; 0],
    x_eq = [0; 0],
    f = f,
    l = l,
    maxiters = 100,
    tol = 1e-2,
    β = 0.7,
    α = 0.4,
);

## --------------------------- regulator parameters --------------------------- ##

Qr = Timeseries(t->diagm([10,1]), model.t) # needs to capture X(t)
Rr = Timeseries(t->1e-3, model.t) # needs to capture X(t)
model = merge(model, (
    Qr=Qr,
    Rr=Rr,
));


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

xt = [0;0]
ut = [0]

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


## --------------------------- zero initial trajectory --------------------------- ##
u = Timeseries(t->[0;], model.t)
x = Timeseries(t->[0;0;], model.t)
ξ = (x,u)


## --------------------------- or, solve zero input dynamics --------------------------- ##
# function dynamics!(dx, x, u, t) 
#     dx .= f(x, u(t))
# end

# T = last(model.t)
# x = solve(ODEProblem(dynamics!, model.x0, (0.0, T), u))
# x = Timeseries(t->x(t), model.t)
# ξ = (x,u)

## --------------------------- optimize --------------------------- ##
@time ξ = pronto(ξ, model)


## --------------------------- plot --------------------------- ##
plot_timeseries(model.t, ξ[1], 2)
