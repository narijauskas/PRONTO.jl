using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef


## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct LaneChange <: Model{6,2}
    M::Float64 = 2041    # [kg]     Vehicle mass
    J::Float64 = 4964    # [kg m^2] Vehicle inertia (yaw)
    g::Float64 = 9.81    # [m/s^2]  Gravity acceleration
    Lf::Float64 = 1.56   # [m]      CG distance, front
    Lr::Float64 = 1.64   # [m]      CG distance, back
    μ::Float64 = 0.8     # []       Coefficient of friction
    b::Float64 = 12      # []       Tire parameter (Pacejka model)
    c::Float64 = 1.285   # []       Tire parameter (Pacejka model)
    s::Float64 = 30      # [m/s]    Vehicle speed
    kr::SVector{2,Float64} = [0.1,0.1]      # LQR
    kq::SVector{6,Float64} = [1,0,1,0,0,0]  # LQR
    xeq::SVector{6,Float64} = zeros(6)      # equilibrium
end

# define model dynamics
@define_f LaneChange begin
    # sideslip angles
    αf = x[5] - atan((x[2] + Lf*x[4])/s)
    αr = x[6] - atan((x[2] - Lr*x[4])/s)
    # tire forces
    F_αf = μ*g*M*sin(c*atan(b*αf))
    F_αr = μ*g*M*sin(c*atan(b*αr))

    [
        s*sin(x[3]) + x[2]*cos(x[3])
        -s*x[4] + ( F_αf*cos(x[5]) + F_αr*cos(x[6]) )/M
        x[4]
        ( F_αf*cos(x[5])*Lf - F_αr*cos(x[6])*Lr )/J
        u[1]
        u[2]
    ]
end

@define_l LaneChange 1/2*(x-xeq)'*I*(x-xeq) + 1/2*u'*I*u
@define_m LaneChange 1/2*(x-xeq)'*I*(x-xeq)
@define_R LaneChange diagm(kr)
@define_Q LaneChange diagm(kq)
resolve_model(LaneChange)
PRONTO.preview(θ::LaneChange, ξ) = ξ.x

## ----------------------------------- solve the problem ----------------------------------- ##

θ = LaneChange(xeq = [1,0,0,0,0,0], kq=[1,0,1,0,0,0])
t0,tf = τ = (0,4)
x0 = SVector{6}(-5.0, zeros(5)...)
xf = @SVector zeros(6)
μ = t->zeros(2)
η = open_loop(θ,x0,μ,τ)
ξ, data = pronto(θ,x0,η,τ; tol=1e-4)
@time ξ, data = pronto(θ,x0,η,τ; tol=1e-4, show_preview=false, show_info=false);
