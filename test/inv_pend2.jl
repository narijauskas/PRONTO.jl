# to compare against PRONTO for Dummies
import Pkg; Pkg.activate(".")
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
# using DataInterpolations
# using DifferentialEquations
using MatrixEquations

using GLMakie; plot(rand(10))
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
    maxiters = 10,
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

function plot_trajectory!(ax, t, X, n; kw...)
    for i in 1:n
        lines!(ax, t, map(τ->X(τ)[i], t); kw...)
    end
    return ax
end

function plot_trajectory(t, X, n; kw...)
    fig = Figure(; kw...)
    ax = Axis(fig[1,1]; kw...)

    plot_trajectory!(ax, t, X, n; kw...)
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
ξ = pronto(ξ, model)





## --------------------------- regulator --------------------------- ##

# set x_eq, u_eq
X.u[:,end] .= x_eq = [0.0,0.0]

# % LQR stage costs
# Pr(x)
# Qr(x,u,t)
# Rr(x,u,t)
Qr = t->diagm([10,1]) # needs to capture X(t)
Rr = t->1e-3 # needs to capture X(t)


Kr,Pt = regulator(X, U, t, Rr, Qr, fx, fu);
display(plot_trajectory(t, Kr, 2))

# % Regulator equilibrium
# reg.xr = @(xT) 0*xT;
# reg.ur = @(xT) [0 0]*xT;


# anonymous = 2.4ms to build, 1μs to execute
# regular function = 2.4ms to build, 1μs to execute
##




## --------------------------- projection --------------------------- ##

X1,U1 = projection(X, U, t, Kr, x0, f);

fig = Figure()
ax = Axis(fig[1,1]; title = "X1"); plot_trajectory!(ax, t, X1, 2)
ax = Axis(fig[2,1]; title = "U1"); plot_trajectory!(ax, t, U1, 1)
display(fig)

# L = cost(X1,U1,t,l)



## --------------------------- tangent space --------------------------- ##


A = t->Main.fx(X1(t), U1(t))
B = t->Main.fu(X1(t), U1(t))
a = t->Main.l_x(X1(t), U1(t))
b = t->Main.l_u(X1(t), U1(t))

fig = Figure()
ax = Axis(fig[1,1]; title = "A"); plot_trajectory!(ax, t, A, 2)
ax = Axis(fig[1,2]; title = "B"); plot_trajectory!(ax, t, B, 1)
ax = Axis(fig[2,1]; title = "a"); plot_trajectory!(ax, t, a, 1)
ax = Axis(fig[2,2]; title = "b"); plot_trajectory!(ax, t, b, 1)
display(fig)

## --------------------------- tangent space cont'd --------------------------- ##


Q = t->Main.lxx(X1(t), U1(t))
R = t->Main.luu(X1(t), U1(t))
S = t->Main.lxu(X1(t), U1(t))

fig = Figure()
ax = Axis(fig[1,1]; title = "Q"); plot_trajectory!(ax, t, Q, 2)
ax = Axis(fig[2,1]; title = "R"); plot_trajectory!(ax, t, R, 1)
ax = Axis(fig[3,1]; title = "S"); plot_trajectory!(ax, t, S, 1)
display(fig)

## --------------------------- search direction --------------------------- ##
# include("search_direction.jl")


# compute q
# check if posdef R
# newton method or gradient descent
# get Ko,vo,Ro,Qo,So


# for now:
Ko,vo,q,z,v,y,Dh,D2g = PRONTO.search_direction(X1, U1, t, model, Kr, zeros(2));

v = PRONTO.tau(τ->v(τ),t);

fig = Figure()
ax = Axis(fig[1,1]; title = "Ko"); plot_trajectory!(ax, t, Ko, 2)
ax = Axis(fig[2,1]; title = "vo"); plot_trajectory!(ax, t, vo, 1)
ax = Axis(fig[3,1]; title = "q"); plot_trajectory!(ax, t, q, 2)
display(fig)


## --------------------------- search direction --------------------------- ##

fig = Figure()
ax = Axis(fig[1,1]; title = "z"); plot_trajectory!(ax, t, z, 2)
ax = Axis(fig[2,1]; title = "v"); plot_trajectory!(ax, t, v, 1)
ax = Axis(fig[3,1]; title = "y"); plot_trajectory!(ax, t, y, 2)
display(fig)

# ζ = (z,v)

## --------------------------- armijo --------------------------- ##

γ = PRONTO.armijo_backstep(X1,U1,t,z,v,Kr,x0,f,l,p,Dh)

## --------------------------- next estimate --------------------------- ##

α = PRONTO.tau(t->(X1(t) + γ*z(t)), t);
μ = PRONTO.tau(t->(U1(t) + γ*v(t)), t);
X2,U2 = projection(α, μ, t, Kr, x0, f);
fig = Figure()
ax = Axis(fig[1,1]; title = "X1"); plot_trajectory!(ax, t, X1, 2)
ax = Axis(fig[2,1]; title = "U1"); plot_trajectory!(ax, t, U1, 1)
ax = Axis(fig[1,2]; title = "X2"); plot_trajectory!(ax, t, X2, 2)
ax = Axis(fig[2,2]; title = "U2"); plot_trajectory!(ax, t, U2, 1)
display(fig)









## --------------------------- pronto loop --------------------------- ##

X0 = X; U0 = U

for iter in 1:200
    @show iter
    Kr,Pt = regulator(X0, U0, t, Rr, Qr, fx, fu);
    X1,U1 = projection(X0, U0, t, Kr, x0, f);
    Ko,vo,q,z,v,y,Dh,D2g = PRONTO.search_direction(X1, U1, t, model, Kr, zeros(2));

    # end condition - always error
    @show Dh
    Dh > 0 ? (@error "increased cost from update direction"; break) : nothing
    -Dh < 1e-2 ? (@error "converged - this is good"; break) : nothing

    v = PRONTO.tau(τ->v(τ),t);
    γ = PRONTO.armijo_backstep(X1,U1,t,z,v,Kr,x0,f,l,p,Dh)
    α = PRONTO.tau(t->(X1(t) + γ*z(t)), t);
    μ = PRONTO.tau(t->(U1(t) + γ*v(t)), t);
    X0,U0 = projection(α, μ, t, Kr, x0, f);
end

fig = Figure()
ax = Axis(fig[1,1]; title = "X (opt)"); plot_trajectory!(ax, t, X0, 2)
ax = Axis(fig[2,1]; title = "U (opt)"); plot_trajectory!(ax, t, U0, 1)
display(fig)







## --------------------------- ?? --------------------------- ##


fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, t, map(τ->X1(τ)[1], t))
lines!(ax, t, map(τ->X1(τ)[2], t))
display(fig)

##

swapdims(x) = permutedims(x, collect)

Ro = R
Ro += q*f


##
S₀ = t -> S(t) .+ sum(map(k->q(τ)[k]*Main.fuu(x(τ),u(τ))[k,:,:], 1:length(q(τ))))

S₀ = τ -> S(τ) .+ sum(map( 1:length(q(τ)) ) do k
    q(τ)[k]*fuu(x(τ), u(τ))[k,:,:]
end)
# end


## --------------------------- plot --------------------------- ##

N = 2
Y = Ko

fig = Figure(); ax = Axis(fig[1,1])
for i in 1:N
    lines!(ax, t, map(τ->Y(τ)[i], t))
end
display(fig)

## --------------------------- # --------------------------- ##


## ---------------------------  --------------------------- ##


# R₀ = R
# Q₀ = Q
# S₀ = S

# R₀ = t -> R(t) .+ mapreduce((qk,fk)->qk*fk, sum, q(t), fuu(X1(t), U1(t)))

# R₀ = t -> R(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fuu(X1(t), U1(t))))
# Q₀ = t -> Q(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxx(X1(t), U1(t))))
# S₀ = t -> S(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxu(X1(t), U1(t))))










