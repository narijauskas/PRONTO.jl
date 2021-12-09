# using Zygote
using ForwardDiff
using ForwardDiff: jacobian
using LinearAlgebra
using LinearAlgebra: I



m = 1
l = 1
g = 9.81

# ---------------- pendulum jacobian ---------------- #

function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = (1/(m*l^2))*(p[1] - m*g*l*sin(x[1]))
end


dx = zeros(2)
x = zeros(2)
M = t->[sin(t)]
X = t->[1, 0]


Jx = t->jacobian((dx,x)->pendulum!(dx, x, M(t), t), dx, X(t))
Jx(0.0)

Jx = t->jacobian((dx,p)->pendulum!(dx, X(t), p, t), dx, M(t))
Jx(0.0)






# ---------------- pendulum jacobian ---------------- #

#YO: assume the following form:
# fxn!(ẋ, x, u, t)
# where x = X(t) and u = U(t)

function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = (1/(m*l^2))*(p(t)[1] - m*g*l*sin(x[1]))
end


dx = zeros(2)
x = zeros(2)


U = t->[sin(t)]
X = t->[1, 0]


Jx = t->jacobian((dx,x)->pendulum!(dx, x, M(t), t), dx, X(t))
Jx(0.0)

Ju = t->jacobian((dx,p)->pendulum!(dx, X(t), M, t), dx, M(t))
Ju(0.0)
















# ---------------- pendulum jacobian ---------------- #

fxn = (x, u) -> [x[2]; (1/(m*l^2))*(u[1] - m*g*l*sin(x[1]))]
Xe = (t)->[sin(t); cos(t)]
Ue = (t)->[m*g*l*sin(Xe(t)[1])]
AB = (t)->jacobian((x,u)->(fxn(x,u)), Xe(t), Ue(t))
A = (t)->AB(t)[1]
B = (t)->AB(t)[2]

A(0)
B(0)


# ---------------- pendulum hessian ---------------- #

f1 = (x,u)->x[2]
f2 = (x,u)->(1/(m*l^2))*(u[1] - m*g*l*sin(x[1]))


























dsin(x) = (sin(x), ȳ -> (ȳ * cos(x),))

f = [xu->xu[2], xu->(xu[3]-m*g*l*sin(xu[1]))/(m*l^2)]

[hessian(fk, [π/2,0,0])[1:2,1:2] for fk in f]


f = xu->(xu.u-m*g*l*sin(xu.x[1]))/(m*l^2)

y, back = pullback(f, args...)
grad = back(sensitivity(y))


y, back = pullback(sin, 0)












f(x,u)
A,B = jacobian((x,u)->(f(x,u)), [1,2], [1,2])

f = x->x
Xe = (t)->[sin(t); cos(t)]

# hessian_dual(f, x::AbstractArray) = forward_jacobian(x -> gradient(f, x)[1], x)[2]


xs = [1,2,3]
jacobian(()->jacobian(()->xs, Params([xs]))[xs], Params([xs]))







# does not work...
f = x->sin.(x)
fx = x->jacobian(f,x)[1]
jacobian(fx, 1)


fx = x->gradient(f,x)
fx([1,2])


H = (t)->hessian(x->f(x), Xe(t))
H(0)

J = jacobian((x)->f(x), Xe(0.0))
A = (t)->jacobian((x)->f(x), Xe(t))
B = (t)->[jacobian((x)->AA(x), Xe(t)) for AA in A(t)]



f(x,u) = x + u


A = (t)->jacobian((x,u)->(f(x,u)), Xe(t), Ue(t))[1]
B = (t)->jacobian((x,u)->(f(x,u)), Xe(t), Ue(t))[2]


AB = (t)->jacobian((x,u)->(f(x,u)), Xe(t), Ue(t))
# AB = (t)->(sin(t), cos(t))
A = (t)->AB(t)[1]
B = (t)->AB(t)[2]

gradient((x,u)->(f(x,u)[1]), 1, 1)


M = t->0.1sin(t)
sol = solve(ODEProblem(pendulum!, [π, 0.0], (0.0,10.0), M), Tsit5())

sol = solve(ODEProblem(pendulum!, [π, 0.0], (10.0,0.0), M), Tsit5())


# a translation/analogue of drive_ipend.m

# set up a function space:
dt = 0.01; t0 = 0.0; t1 = 10.0
T0 = t0:dt:t1

Φ₀ = @. tanh(T0-t1/2)*(1-tanh(T0-t1/2)^2)
maximum(Φ₀)
