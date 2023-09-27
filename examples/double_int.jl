using PRONTO
using LinearAlgebra
using MatrixEquations
using StaticArrays
using Base: @kwdef

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct DoubleInt <: Model{2,1}
    R::Float64 
    Q::SMatrix{2,2,Float64}
    P::SMatrix{2,2,Float64}
end

A = [0 1; 0 0]
B = [0; 1]

@define_f DoubleInt A*x + B*u[1]

@define_l DoubleInt 1/2*R*u[1]^2 + 1/2*x'*Q*x

@define_m DoubleInt 1/2*x'*P*x



@define_Qr DoubleInt I(2)
@define_Rr DoubleInt I(1)

resolve_model(DoubleInt)
PRONTO.preview(θ::DoubleInt, ξ) = ξ.x

## ----------------------------------- solve the problem ----------------------------------- ##

R = 0.04
Q = diagm([1.0, 0.0])
P = arec(A,B,R*I,Q)[1]
θ = DoubleInt(R, Q, P) 
τ = t0,tf = 0,2
x0 = @SVector [2,0]

μ = t->[0]

η = open_loop(θ,x0,μ,τ)

ξ,data = pronto(θ,x0,η,τ; tol=1e-6);

## ----------------------------------- plot the results ----------------------------------- ##

import Pkg
Pkg.activate()
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "position [m]")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "velocity [m/s]")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "acceleration [m/s²]")

lines!(ax1, ts, [ξ.x(t)[1] for t in ts], color = :blue, linewidth = 2)
lines!(ax2, ts, [ξ.x(t)[2] for t in ts], color = :green, linewidth = 2)
lines!(ax3, ts, [ξ.u(t)[1] for t in ts], color = :red, linewidth = 2)

display(fig)