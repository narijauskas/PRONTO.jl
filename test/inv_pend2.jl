# to compare against PRONTO for Dummies
import Pkg; Pkg.activate(".")
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
using MatrixEquations
using DataInterpolations

# using GLMakie; display(lines(rand(10)))
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
nx = 2
nu = 1
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

## --------------------------- regulator parameters --------------------------- ##

Qr = Timeseries(t->diagm([10,1]), model.t) # needs to capture X(t)
Rr = Timeseries(t->1e-3, model.t) # needs to capture X(t)


## --------------------------- build model --------------------------- ##

model = (
    t = 0:0.001:10,
    x0 = [2π/3; 0],
    x_eq = [0; 0],
    f = f,
    l = l,
    maxiters = 100,
    tol = 1e-3,
    β = 0.7,
    α = 0.4,
    Qr=Qr,
    Rr=Rr,
);




## --------------------------- autodiff & terminal cost --------------------------- ##
include("build_model.jl");


## --------------------------- zero initial trajectory --------------------------- ##
u0 = Timeseries(t->[0;], model.t)
x0 = Timeseries(t->[0;0;], model.t)
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
plot_timeseries(model.t, ξ[1], 2)
