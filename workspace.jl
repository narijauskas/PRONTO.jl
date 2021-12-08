using Zygote
using LinearAlgebra
using LinearAlgebra: I

f(x,u)
A,B = jacobian((x,u)->(f(x,u)), [1,2], [1,2])

f(x) = x
Xe(t) = [sin(t); cos(t)]
A = (t)->jacobian((x)->f(x), Xe(t))




f(x,u) = x + u


A = (t)->jacobian((x,u)->(f(x,u)), Xe(t), Ue(t))[1]
B = (t)->jacobian((x,u)->(f(x,u)), Xe(t), Ue(t))[2]


AB = (t)->jacobian((x,u)->(f(x,u)), Xe(t), Ue(t))
# AB = (t)->(sin(t), cos(t))
A = (t)->AB(t)[1]
B = (t)->AB(t)[2]

gradient((x,u)->(f(x,u)[1]), 1, 1)

m = 1
l = 1
g = 9.81

dx[1] = x[2]
dx[2] = (1/(m*l^2))*(u[1] - m*g*l*sin(x[1]))

fxn = (x, u) -> [x[2]; (1/(m*l^2))*(u[1] - m*g*l*sin(x[1]))]
Xe = (t)->[sin(t); cos(t)]
Ue = (t)->[m*g*l*sin(Xe(t)[1])]
AB = (t)->jacobian((x,u)->(fxn(x,u)), Xe(t), Ue(t))
A = (t)->AB(t)[1]
B = (t)->AB(t)[2]



# a translation/analogue of drive_ipend.m

# set up a function space:
dt = 0.01; t0 = 0.0; t1 = 10.0
T0 = t0:dt:t1

Φ₀ = @. tanh(T0-t1/2)*(1-tanh(T0-t1/2)^2)
maximum(Φ₀)
