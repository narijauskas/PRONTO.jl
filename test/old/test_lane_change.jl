# using Test
using PRONTO
using LinearAlgebra, StaticArrays
using Base: @kwdef


NX = 6; NU = 2
@kwdef struct LaneChange3 <: PRONTO.Model{6,2,17}
    M::Float64 = 2041    # [kg]     Vehicle mass
    J::Float64 = 4964    # [kg m^2] Vehicle inertia (yaw)
    g::Float64 = 9.81    # [m/s^2]  Gravity acceleration
    Lf::Float64 = 1.56   # [m]      CG distance, front
    Lr::Float64 = 1.64   # [m]      CG distance, back
    μ::Float64 = 0.8     # []       Coefficient of friction
    b::Float64 = 12      # []       Tire parameter (Pacejka model)
    c::Float64 = 1.285   # []       Tire parameter (Pacejka model)
    s::Float64 = 30      # [m/s]    Vehicle speed
    r1::Float64 = 0.1    # LQR
    r2::Float64 = 0.1    # LQR
    q1::Float64 = 1      # LQR
    q2::Float64 = 0      # LQR
    q3::Float64 = 1      # LQR
    q4::Float64 = 0      # LQR
    q5::Float64 = 0      # LQR
    q6::Float64 = 0      # LQR
    # kr::SVector{2} = [0.1,0.1]      # LQR
    # kq::SVector{6} = [1,0,1,0,0,0]  # LQR
    # xeq::SVector{6} = zeros(6)      # equilibrium
end

# sideslip angles
αf(x,θ) = x[5] - atan((x[2] + θ.Lf*x[4])/θ.s)
αr(x,θ) = x[6] - atan((x[2] - θ.Lr*x[4])/θ.s)

# tire force
F(α,θ) = θ.μ*θ.g*θ.M*sin(θ.c*atan(θ.b*α))

# define model dynamics
function dynamics(x,u,t,θ)
    [
        θ.s*sin(x[3]) + x[2]*cos(x[3]),
        -θ.s*x[4] + ( F(αf(x,θ),θ)*cos(x[5]) + F(αr(x,θ),θ)*cos(x[6]) )/θ.M,
        x[4],
        ( F(αf(x,θ),θ)*cos(x[5])*θ.Lf - F(αr(x,θ),θ)*cos(x[6])*θ.Lr )/θ.J,
        u[1],
        u[2],
    ]
end

stagecost(x,u,t,θ) = 1/2*collect(x')I*x + 1/2*collect(u')I*u

# should be solution to DARE at desired equilibrium
# ref = zeros(NX)
# Pl,_ = ared(fx(ref,zeros(NU)), fu(ref,zeros(NU)), Rlqr, Qlqr)
termcost(x,u,t,θ) = 1/2*collect(x')*x

regR(x,u,t,θ) = diagm([θ.r1,θ.r2])
regQ(x,u,t,θ) = diagm([θ.q1,θ.q2,θ.q3,θ.q4,θ.q5,θ.q6])


# regP(x,u,t,θ) = 

PRONTO.generate_model(LaneChange3, dynamics, stagecost, termcost, regQ, regR)



## -------------------------------  ------------------------------- ##
θ = LaneChange(s=15)
x0 = SVector{6}(-5.0,zeros(5)...)
xf = @SVector zeros(6)
t0,tf = τ = (0,10)

μ = @closure t->SVector{2}(zeros(2))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50)


## -------------------------------  ------------------------------- ##





struct Yeet <: FieldVector{4,Float64}
    a::Float64
    b::SVector{3,Float64}
end

yeet = Yeet(1,[2,3,4])


@kwdef struct Yop <: FieldVector{2,Float64}
    a::Float64
    b::SVector{3,Float64} = [0.1,0.1,0.1]
end
yop = Yop(1,[2,3,4])

yop = Yop(a=1)






# regulator
Rr = (θ,t,x,u) -> 0.1*diagm([1,1])
Qr = (θ,t,x,u) -> 1.0*diagm([1,0,1,0,0,0])

let

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
    f = (θ,t,x,u) -> [

            s*sin(x[3]) + x[2]*cos(x[3]),
            -s*x[4] + ( F(αf(x))*cos(x[5]) + F(αr(x))*cos(x[6]) )/M,
            x[4],
            ( F(αf(x))*cos(x[5])*Lf - F(αr(x))*cos(x[6])*Lr )/J,
            u[1],
            u[2],
        ]


    # define stage cost


    # regulator
    Rr = (θ,t,x,u) -> 0.1*diagm([1,1])
    Qr = (θ,t,x,u) -> 1.0*diagm([1,0,1,0,0,0])

    # template
    # f = (θ,t,x,u) ->
    # l = (θ,t,x,u) ->
    # p = (θ,t,x,u) ->
    # Rr = (θ,t,x,u) ->
    # Qr = (θ,t,x,u) ->

    @derive LaneChange
end




## ----------------------------------- tests ----------------------------------- ##

M = LaneChange()
θ = Float64[]
t0 = 0.0
tf = 10.0
x0 = [-5.0;zeros(nx(M)-1)]
xf = zeros(nx(M))
u0 = zeros(nu(M))
uf = zeros(nu(M))

ξ0 = vcat(x0,u0)

##
φg = @closure t->[smooth(t,x0,xf,tf); 0.1*ones(nu(M))]
φ = guess_φ(M,θ,ξ0,t0,tf,φg)
@time ξ = pronto(M,θ,t0,tf,x0,u0,φ; tol = 1e-4)

##

model = (
    ts = 0:0.001:10,

    x0 = [-5.0;zeros(NX-1)],
    tol = 1e-4,
    x_eq = zeros(NX),
    u_eq = zeros(NU),
    maxiters = 20,
    α = 0.4,
    β = 0.7,
)

#TEST: move inside f(x,u)


η = pronto(model)
ts = model.ts


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

