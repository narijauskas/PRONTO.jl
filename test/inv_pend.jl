# to compare against PRONTO for Dummies
import Pkg; Pkg.activate(".")
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using MatrixEquations

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
Ql = I
Rl = I
l(x,u) = 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)

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


## --------------------------- build estimate --------------------------- ##
# solve dynamics with zero input

T = 10
t = 0:0.001:T
x0 = [2π/3; 0]

U = LinearInterpolation(zeros(1, length(t)), t)
X = LinearInterpolation(zeros(2, length(t)), t)
display(plot_trajectory(t, X, 2))


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


## --------------------------- solve dynamics --------------------------- ##

function dynamics!(dx, x, u, t) 
    dx .= f(x, u(t))
end

sln = solve(ODEProblem(dynamics!, x0, (0.0, T), U))
Xt = LinearInterpolation(hcat(sln.(t)...), t) 
display(plot_trajectory(t, Xt, 2))


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










