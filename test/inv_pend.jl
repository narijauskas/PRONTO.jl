# to compare against PRONTO for Dummies
import Pkg; Pkg.activate(".")
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using MatrixEquations

using GLMakie; #plot(rand(10))

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
Q = I
R = I
l(x,u) = 1/2*collect(x)'*Q*collect(x) + 1/2*collect(u)'*R*collect(u)

# temporary workaround, until Symbolics gets full array variable support
# l(x,u) = 1/2*(x[1]^2 + x[2]^2 + u[1]^2)



## --------------------------- autodiff --------------------------- ##

fx = jacobian(x, f, x, u)
fu = jacobian(u, f, x, u)

fxx = jacobian(x, fx, x, u)
fxu = jacobian(u, fx, x, u)
fux = jacobian(x, fu, x, u)
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
    :fxx => fxx,
    :fxu => fxu,
    :fuu => fuu,
    :l => l,
    :lx => l_x,
    :lu => l_u,
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
# Pr(x)
# Qr(x,u,t)
# Rr(x,u,t)
Qr = t->diagm([10,1]) # needs to capture X(t)
Rr = t->1e-3 # needs to capture X(t)


Kr,Pt = regulator(X, U, t, Rr, Qr, fx, fu);
# % Regulator equilibrium
# reg.xr = @(xT) 0*xT;
# reg.ur = @(xT) [0 0]*xT;


# anonymous = 2.4ms to build, 1μs to execute
# regular function = 2.4ms to build, 1μs to execute (but, has nasty typedef)
##
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, t, map(τ->Kr(τ)[1], t))
lines!(ax, t, map(τ->Kr(τ)[2], t))
display(fig)
# α = 
# μ = 



## --------------------------- projection --------------------------- ##

X1,U1 = projection(X, U, t, Kr, x0, f);


##
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, t, map(τ->X1(τ)[1], t))
lines!(ax, t, map(τ->X1(τ)[2], t))

ax = Axis(fig[2,1])
lines!(ax, t, map(τ->X1(τ)[1], t))
display(fig)

# L = cost(X1,U1,t,l)



## --------------------------- tangent space --------------------------- ##


A = t->Main.fx(X1(t), U1(t))
B = t->Main.fu(X1(t), U1(t))
a = t->Main.l_x(X1(t), U1(t))
b = t->Main.l_u(X1(t), U1(t))
Q = t->Main.lxx(X1(t), U1(t))
R = t->Main.luu(X1(t), U1(t))
S = t->Main.lxu(X1(t), U1(t))


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
ax = Axis(fig[1,1])
lines!(ax, t, map(τ->Ko(τ)[1], t))
lines!(ax, t, map(τ->Ko(τ)[2], t))

ax = Axis(fig[2,1])
lines!(ax, t, map(τ->vo(τ)[1], t))
display(fig)

ax = Axis(fig[3,1])
lines!(ax, t, map(τ->q(τ)[1], t))
lines!(ax, t, map(τ->q(τ)[2], t))

display(fig)

## --------------------------- search direction --------------------------- ##

fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, t, map(τ->z(τ)[1], t))
lines!(ax, t, map(τ->z(τ)[2], t))

ax = Axis(fig[2,1])
lines!(ax, t, map(τ->v(τ)[1], t))
display(fig)

## --------------------------- search direction --------------------------- ##
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, t, map(τ->y(τ)[1], t))
lines!(ax, t, map(τ->y(τ)[2], t))
display(fig)

# R₀ = R
# Q₀ = Q
# S₀ = S

# R₀ = t -> R(t) .+ mapreduce((qk,fk)->qk*fk, sum, q(t), fuu(X1(t), U1(t)))

# R₀ = t -> R(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fuu(X1(t), U1(t))))
# Q₀ = t -> Q(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxx(X1(t), U1(t))))
# S₀ = t -> S(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxu(X1(t), U1(t))))











# ζ = (z,v)

## --------------------------- next estimate --------------------------- ##

γ = PRONTO.armijo_backstep(X1,U1,t,z,v,Kr,x0,f,l,p,Dh)



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
