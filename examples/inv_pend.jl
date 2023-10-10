using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct InvPend <: Model{2,1}
    L::Float64 = 2 
    g::Float64 = 9.81 
    ρ::Float64 = 1;
end

@define_f InvPend [
    x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
]
@define_l InvPend 1/2*ρ*u[1]^2
@define_m InvPend 1-cos(x[1])+x[2]^2/2

@define_Qr InvPend diagm([10, 1])
@define_Rr InvPend diagm([1e-3])

resolve_model(InvPend)
PRONTO.preview(θ::InvPend, ξ) = ξ.x

## ----------------------------------- solve the problem ----------------------------------- ##

θ = InvPend() 
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]

μ = t->[0]
α = t->[0;0]

η = closed_loop(θ,x0,α,μ,τ)

ξ,data = pronto(θ,x0,η,τ; tol=1e-3);

## ----------------------------------- plot the results ----------------------------------- ##
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time [s]", ylabel = "angular position [rad]")
ax2 = Axis(fig[2,1], xlabel = "time [s]", ylabel = "angular velocity [rad/s]")
ax3 = Axis(fig[3,1], xlabel = "time [s]", ylabel = "control input [m/s^2]")

lines!(ax1, ts, [ξ.x(t)[1] for t in ts], color = :blue, linewidth = 2)
lines!(ax2, ts, [ξ.x(t)[2] for t in ts], color = :green, linewidth = 2)
lines!(ax3, ts, [ξ.u(t)[1] for t in ts], color = :red, linewidth = 2)

display(fig)