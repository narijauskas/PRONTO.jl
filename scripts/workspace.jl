# using Zygote
using ForwardDiff
using ForwardDiff: gradient, jacobian, derivative
using LinearAlgebra
using LinearAlgebra: I
using QuadGK
using PRONTO
##

Jx(f, X, U) = jacobian(x->f(x, U), X)
A = t->Jx(pendulum, X(t), U(t))

m=g=l=1

pendulum(x, u) = [
    x[2],
    (1/(m*l^2))*(u[1] - 2*m*g*l*sin(x[1]))
]



U = t->[cos(t)]
X = t->[π/2, 0]


A(1.0)








f = g->quadgk(g, 0, π)[1]

f(sin)

fg = t->derivative(f, sin(t)) # where f(x)

fg(1)














m = 1
l = 1
g = 9.81

# ---------------- pendulum jacobian ---------------- #



function pendulum(x, u)
    dx = similar(x)
    pendulum!(dx, x, u)
end



# for compatibility with OrdinaryDiffEq
# ode_pendulum!(dx, x, p, t) = pendulum!(dx, x, p(t))

U = t->[cos(t)]
X = t->[π/2, 0]


function Jx(f!, X, U)
    return t->jacobian(similar(X(0)), X(t)) do dx, x
        f!(dx, x, U(t))
    end
end

# flex:
# Jx(f!, X, U) = t->jacobian((dx, x)->f!(dx, x, U(t)), similar(X(0)), X(t))


function Ju(f!, X, U)
    return t->jacobian(similar(X(0)), U(t)) do dx, u
        f!(dx, X(t), u)
    end
end

A = Jx(pendulum!, X, U)
B = Ju(pendulum!, X, U)

A(0)
B(0)
# ---------------- pendulum hessian ---------------- #


# function vector_hessian(f, x)
#        n = length(x)
#        out = jacobian(x -> jacobian(f, x), x)
#     return reshape(out, n, n, n)
# end

# test_pendulum!(dx, x) = pendulum!(dx, x, [1])

# function test_pendulum(x)
#     return [x[2], (1/(m*l^2))*(1 - m*g*l*sin(x[1]))]
# end

# jacobian(x->jacobian(test_pendulum, x), [π/2, 0])


# jacobian((ddx,x) -> jacobian!(ddx, test_pendulum!, zeros(2), x), zeros(2, 2), [π/2, 0])

# # jacobian(v->jacobian(f!, ), X(t))




function Hxx(f!, X, U)
    nx = length(X(0))
    function f(x, u)
        dx = Array{promote_type(eltype(x), eltype(u))}(undef, size(x)...)
        f!(dx, x, u)
    end
    hess = t->jacobian(X(t)) do xx
        jacobian(xx) do x
            f(x, U(t))
        end
    end
    return t->permutedims(reshape(hess(t), nx, nx, nx), (2,3,1))
end


function Huu(f!, X, U)
    nu = length(U(0))
    nx = length(X(0))
    function f(x, u)
        dx = Array{promote_type(eltype(x), eltype(u))}(undef, size(x)...)
        f!(dx, x, u)
    end
    hess = t->jacobian(U(t)) do uu
        jacobian(uu) do u
            f(X(t),u)
        end
    end
    return t->permutedims(reshape(hess(t), nx, nu, nu), (2,3,1))
end


function Hxu(f!, X, U)
    nu = length(U(0))
    nx = length(X(0))
    function f(x, u)
        dx = Array{promote_type(eltype(x), eltype(u))}(undef, size(x)...)
        f!(dx, x, u)
    end
    hess = t->jacobian(U(t)) do u
        jacobian(x->f(x, u), X(t))
    end
    return t->permutedims(reshape(hess(t), nx, nu, nx), (2,3,1))
end


yeet = Hxu(pendulum!, X, U)
yeet(0)

# jacobian(x -> jacobian((dx,x)->pendulum!(dx, x, U(t)), x), X(0))


yop = t->[sin(t), cos(t), cos(t), sin(t)]
yeet = t->reshape(yop(t), 2, 2)

@btime yeet(0)









# ---------------- non-mutating jacobian/hessian ---------------- #


function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = (1/(m*l^2))*(u[1] - m*g*l*sin(x[1]))
    return dx
end


pendulum(x, u) = [
    x[2],
    (1/(m*l^2))*(u[1] - 2*m*g*l*sin(x[1]))
]


U = t->[cos(t)]
X = t->[sin(t), 0]

Jx(f, X, U) = t->jacobian(x->f(x, U(t)), X(t))
Ju(f, X, U) = t->jacobian(u->f(X(t), u), U(t))


function Hxx(f, X, U)
    nx = length(X(0))
    hess = t->jacobian(X(t)) do xx
        jacobian(x->f(x, U(t)), xx)
    end
    return t->permutedims(reshape(hess(t), nx, nx, nx), (2,3,1))
end


function Huu(f, X, U)
    nu = length(U(0))
    nx = length(X(0))
    hess = t->jacobian(U(t)) do uu
        jacobian(u->f(X(t), u), uu)
    end
    return t->permutedims(reshape(hess(t), nx, nu, nu), (2,3,1))
end


function Hxu(f, X, U)
    nu = length(U(0))
    nx = length(X(0))
    hess = t->jacobian(U(t)) do u
        jacobian(x->f(x, u), X(t))
    end
    return t->permutedims(reshape(hess(t), nx, nu, nx), (2,3,1))
end


Hxx(pendulum, X, U)(0.0)
Hxu(pendulum, X, U)(0.0)
Huu(pendulum, X, U)(0.0)


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

## cost function testing
ξd = Trajectory(t ->  [0.0;0.0], t -> [0.0;0.0])
Q = identity(2)
R = identity(2)
P1 = identity(2)
T = 10
l, m = build_LQ_cost(ξd, Q, R, P1, T)