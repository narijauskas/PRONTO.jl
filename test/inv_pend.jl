# to compare against PRONTO for Dummies
using Revise, BenchmarkTools
using Symbolics
using LinearAlgebra


using PRONTO
using PRONTO: jacobian, regulator

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
# temporary workaround, while Symbolics gets array variable support
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

X = LinearInterpolation(zeros(2, length(t)), t) 
U = LinearInterpolation(zeros(1, length(t)), t)




## --------------------------- regulator --------------------------- ##

x0 = [2π/3; 0]

# % LQR stage costs
# reg.Qr = diag([10 1]);
# reg.Rr = 1e-3;

# % Regulator equilibrium
# reg.xr = @(xT) 0*xT;
# reg.ur = @(xT) [0 0]*xT;

# α = 
# μ = 