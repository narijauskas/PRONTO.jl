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
    tol = 1e-4,
    x_eq = zeros(NX),
    u_eq = zeros(NU),
    maxiters = 20,
    α = 0.4,
    β = 0.7,
)

#TEST: move inside f(x,u)
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

# define model dynamics
function f(x,u)
    return [
        s*sin(x[3]) + x[2]*cos(x[3]),
        -s*x[4] + ( F(αf(x))*cos(x[5]) + F(αr(x))*cos(x[6]) )/M,
        x[4],
        ( F(αf(x))*cos(x[5])*Lf - F(αr(x))*cos(x[6])*Lr )/J,
        u[1],
        u[2],
    ]
end

# define stage cost
Ql = I
Rl = I

#NOTE: need to collect() vector variables
l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)

# define terminal cost
p = (x) -> 1/2*collect(x)'collect(x)

@info "running autodiff"
model = autodiff(model,f,l,p)
@info "autodiff complete"

model = merge(model, (
    Qr = Interpolant(t->1.0*diagm([1,0,1,0,0,0]), model.ts, NX, NX),
    Rr = Interpolant(t->0.1*diagm([1,1]), model.ts, NU, NU),
    iRr = Interpolant(t->inv(0.1*diagm([1,1])), model.ts, NU, NU),
))


tx = map(1:10) do i
    @elapsed begin
        η = pronto(model)
    end
end


for i in 1:10
    tx = @elapsed begin
        η = pronto(model)
    end
    println("solve time: $tx seconds")
end
# ts = model.ts


#= plot result
    using GLMakie
    fig = Figure()
    ax = Axis(fig[1,1])
    for i in 1:NX
        lines!(ax, ts, map(t->η[1](t)[i], ts))
    end
    ax = Axis(fig[2,1])
    for i in 1:NU
        lines!(ax, ts, map(t->η[2](t)[i], ts))
    end
    display(fig)
=#

