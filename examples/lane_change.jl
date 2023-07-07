# using Test
using PRONTO
using LinearAlgebra, StaticArrays

# NX = 6; NU = 2
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

# sideslip angles
# αf(x,Lf,s) = x[5] - atan((x[2] + Lf*x[4])/s)
# αr(x,Lf,s) = x[6] - atan((x[2] - Lr*x[4])/s)

# # tire force
# F(α,θ) = μ*g*M*sin(c*atan(b*α))

# define model dynamics
@dynamics LaneChange begin
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
# m should be solution to DARE at desired equilibrium
@define_m LaneChange 1/2*(x-xeq)'*I*(x-xeq)
@define_R LaneChange diagm(kr)
@define_Q LaneChange diagm(kq)
resolve_model(LaneChange)
PRONTO.runtime_info(θ::LaneChange, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x, 1; color=PRONTO.manto_colors[1]))


## -------------------------------  ------------------------------- ##
θ = LaneChange(xeq = [1,0,0,0,0,0], kq=[0.1,0,1,0,0,0])
x0 = SVector{6}(-5.0,zeros(5)...)
xf = @SVector zeros(6)
t0,tf = τ = (0,4)
μ = t->zeros(2)
# μ = @closure t->SVector{2}(zeros(2))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ; tol = 1e-6, maxiters = 50);

# plot_lane_change(ξ,τ)
## -------------------------------  ------------------------------- ##


function plot_lane_change(ξ,τ)
    fig = Figure()
    ts = LinRange(τ...,10001)+

    ax = Axis(fig[1,1]; title = "state")
    is = eachindex(ξ.x)
    xs = [ξ.x(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, xs[:,i]), is)

    ax = Axis(fig[2,1]; title = "inputs")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)

    return fig
end
display(fig)







## -------------------------------  ------------------------------- ##
using Base.Threads: @spawn
using ThreadTools

x0 = SVector{6}(-5.0,zeros(5)...)
xf = @SVector zeros(6)
t0,tf = τ = (0,4)

μ = @closure t->SVector{2}(zeros(2))
φ = open_loop(θ,x0,μ,τ)


tk = map([1,2,10]) do s
    θ = LaneChange(s=s)
    @spawn pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, verbose = false)
end

fetch.(tk)

# just the inputs
[ξ.x for ξ in fetch.(tk)]


# plot_lane_change(ξ,τ)
PRONTO.set_plot_scale(15,80)

























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

