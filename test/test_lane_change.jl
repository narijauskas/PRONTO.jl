# using Test
using PRONTO
using LinearAlgebra
using BenchmarkTools
# using JET


NX = 6
NU = 2
model = (
    ts = 0:0.001:10,
    NX = NX,
    NU = NU,
    x0 = [-5.0;zeros(NX-1)],
    tol = 1e-3,
    x_eq = zeros(NX),
    u_eq = zeros(NU),
    maxiters = 20,
    α = 0.4,
    β = 0.7,
)


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
function f(x,u)
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

@info "running autodiff"
model = autodiff(model,f,l,p)
@info "autodiff complete"

model = merge(model, (
    Qr = Interpolant(t->1.0*diagm([1,0,1,0,0,0]), model.ts, NX, NX),
    Rr = Interpolant(t->0.1*diagm([1,1]), model.ts, NU, NU),
    iRr = Interpolant(t->inv(0.1*diagm([1,1])), model.ts, NU, NU),
))


(α,μ) = pronto(model)
ts = model.ts

fig = Figure()
ax = Axis(fig[1,1])
for i in 1:NX
    lines!(ax, ts, map(t->α(t)[i], ts))
end
ax = Axis(fig[2,1])
for i in 1:NU
    lines!(ax, ts, map(t->μ(t)[i], ts))
end
display(fig)


# const X_x = Interpolant(t->model.x0, model.ts, NX)
# const U_u = Interpolant(model.ts, NU)

# # X_x(1.3)
# # @benchmark X_x(1.3)
# update!(t->model.x0, X_x)
# # @benchmark update!(t->model.x0, X_x)


# A = Functor(NX,NX) do buf,t
#     model.fx!(buf, X_x(t), U_u(t))
# end

# A(1.3)
# @benchmark A(1.3)
# @profview for ix in 1:1000000
#     A(1.3)
# end
# @code_warntype A(1.3)
# buf = copy(A.buf)
# @code_warntype model.fx!(buf, X_x(1.3), U_u(1.3))
# @report_opt model.fx!(buf, X_x(1.3), U_u(1.3))
# @report_opt A(1.3)
# @code_warntype A(1.3)


# invRr = Interpolant(t->inv(0.1*diagm([1,1])), model.ts, NU, NU)

# Qr = Functor(NX,NX) do buf,t
#     buf .= 1.0*diagm([1,0,1,0,0,0])
# end

# Rr = Functor(NU,NU) do buf,t
#     buf .= 0.1*diagm([1,1])
# end


    # # can also make these anonymous functions
# Qr = Interpolant(t->1.0*diagm([1,0,1,0,0,0]), model.ts)
# Rr = Interpolant(t->0.1*diagm([1,1]), model.ts)
# invRr = Interpolant(t->inv(Rr(t)), model.ts)

# α0 = Interpolant(t->zeros(model.NX),model.ts)
# μ0 = Interpolant(t->zeros(model.NU),model.ts)


# # x,u = pronto(model,x0,u0)

# using JET
# @time x,u = pronto(model,x0,u0)
# @report_opt x,u = pronto(model,x0,u0)
# @profview x,u = pronto(model,x0,u0)


