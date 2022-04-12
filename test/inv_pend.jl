# to compare against PRONTO for Dummies
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
using DataInterpolations
using DifferentialEquations

using GLMakie; plot(rand(10))

using PRONTO
using PRONTO: jacobian, regulator, projection, cost

## --------------------------- problem definition --------------------------- ##

# symbolic states
@variables x[1:2] u[1:1]


# dynamics
g = 9.81
L = 2
f(x,u) = [x[2];   g/L*sin(x[1])-u[1]*cos(x[1])/L]


# cost function
# Q = collect(I(2))
# R = [1;;]
# l(x,u) = 1/2*x'*Q*x + 1/2*u'*R*u
# temporary workaround, until Symbolics gets array variable support
l(x,u) = 1/2*(x[1]^2 + x[2]^2 + u[1]^2)



## --------------------------- autodiff --------------------------- ##

fx = jacobian(x, f, x, u)
fu = jacobian(u, f, x, u)

fxx = jacobian(x, fx, x, u)
fxu = jacobian(u, fx, x, u)
fuu = jacobian(u, fu, x, u)

l_x = jacobian(x, l, x, u)
l_u = jacobian(u, l, x, u)

lxx = jacobian(x, l_x, x, u)
lxu = jacobian(u, l_x, x, u)
luu = jacobian(u, l_u, x, u)



## --------------------------- build estimate --------------------------- ##
# solve dynamics with zero input

T = 10
t = 0:0.001:T
x0 = [2π/3; 0]

U = LinearInterpolation(zeros(1, length(t)), t)
X = LinearInterpolation(zeros(2, length(t)), t)



A = fx(X(T), U(T))
B = fu(X(T), U(T))
Q = lxx(X(T), U(T))
R = luu(X(T), U(T))
S = lxu(X(T), U(T))
Po,_ = arec(A, B, R, Q ,S)

p(x) = 1/2*collect(x)'*Po*collect(x)
px = jacobian(x, p, x)
pxx = jacobian(x, px, x)
# P = arec(A(T), B(T)inv(R(T))B(T)', Q(T), S(T))


model = Dict(
    :f => f,
    :fx => fx,
    :fu => fu,
    :l => l,
    :l_x => l_x,
    :l_u => l_u,
    :lxx => lxx,
    :lxu => lxu,
    :luu => luu,
    :p => p,
    :px => px,
    :pxx => pxx,
)

# dynamics!(dx, x, u, t) = dx .= f(x, u(t))
# sln = solve(ODEProblem(dynamics!, x0, (0.0, T), U))
# X = LinearInterpolation(hcat(sln.(t)...), t) 


# now we have an estimate trajectory (X,U,t)
# where X(t) and U(t) are callable

## --------------------------- regulator --------------------------- ##

# set x_eq, u_eq
X.u[:,end] .= x_eq = [0.0,0.0]

# % LQR stage costs
Qr = t->diagm([10,1])
Rr = t->1e-3


Kr,Pt = regulator(X, U, t, Rr, Qr, fx, fu);
# % Regulator equilibrium
# reg.xr = @(xT) 0*xT;
# reg.ur = @(xT) [0 0]*xT;


# anonymous = 2.4ms to build, 1μs to execute
# regular function = 2.4ms to build, 1μs to execute (but, has nasty typedef)

fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, t, map(τ->Kr(τ)[1], t))
lines!(ax, t, map(τ->Kr(τ)[2], t))
display(fig)
# α = 
# μ = 


X1,U1 = projection(X, U, t, Kr, x0, f);
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, t, map(τ->X1(τ)[1], t))
lines!(ax, t, map(τ->X1(τ)[2], t))
display(fig)

# L = cost(X1,U1,t,l)




A = t->model[:fx](X1(t), U1(t))
B = t->model[:fu](X1(t), U1(t))
Q = t->model[:lxx](X1(t), U1(t))
R = t->model[:luu](X1(t), U1(t))
S = t->model[:lxu](X1(t), U1(t))


Ko = PRONTO.gradient_descent(X1,U1,t,model);
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, t, map(τ->Ko(τ)[1], t))
lines!(ax, t, map(τ->Ko(τ)[2], t))
display(fig)

