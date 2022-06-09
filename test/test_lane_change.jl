# using Test
using PRONTO
using LinearAlgebra

model = MStruct()
model.ts = 0:0.001:10
model.NX = 6
model.NU = 2
model.x0 = zeros(model.NX)
model.maxiters = 10

# model parameters
M = 2041    # [kg]     Vehicle mass
J = 4964    # [kg m^2] Vehicle inertia (yaw)
g = 9.81    # [m/s^2]  Gravity acceleration
Lf = 1.56   # [m]      CG distance, front
Lr = 1.64   # [m]      CG distance, back
μ = 0.8     # []       Coefficient of friction
b = 12      # []       Tire parameter (Pacejka model)
c = 1.285   # []       Tire parameter (Pacejka model)
s = 30      # [m/s]    Vehicle speed

# sideslip angles
αf(x) = x[5] - atan((x[2] + Lf*x[4])/s)
αr(x) = x[6] - atan((x[2] - Lr*x[4])/s)

# tire force
F(α) = μ*g*M*sin(c*atan(b*α))

# model dynamics
function fxn(x,u)
    # continuous dynamics
    return [
        s*sin(x[3]) + x[2]*cos(x[3]),
        -s*x[4] + ( F(αf(x))*cos(x[5]) + F(αr(x))*cos(x[6]) )/M,
        x[4],
        ( F(αf(x))*cos(x[5])*Lf - F(αr(x))*cos(x[6])*Lr )/J,
        u[1],
        u[2],
    ]
end

# stage cost
Ql = I
Rl = I

l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
p = (x)-> 1/2*collect(x)'collect(x)


PRONTO.autodiff!(model,fxn,l,p)

# can also make these anonymous functions
model.Qr = Interpolant(t->diagm([1,0,1,0,0,0]), model.ts)
model.Rr = Interpolant(t->0.1*diagm([1,1]), model.ts)

u0 = Interpolant(t->zeros(model.NU),model.ts)
x0 = Interpolant(t->zeros(model.NX),model.ts)

x,u = pronto(model,x0,u0)