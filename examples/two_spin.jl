using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef


@kwdef struct TwoSpin <: Model{4,1}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


## --------------------- option 2 --------------------- ##

@define_f TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@define_l TwoSpin begin
    Rl = [0.01;;]
    1/2*u'*Rl*u
end

@define_m TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*x'*Pl*x
end

@define_Q TwoSpin kq*I(4)
@define_R TwoSpin kr*I(1)

PRONTO.Pf(θ::TwoSpin, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))


## plots
import Pkg
Pkg.activate()
using GLMakie, Statistics
Pkg.activate(".")

function plot_spin(ξ,τ)
    fig = Figure()
    ts = LinRange(τ...,10001)

    ax = Axis(fig[1:2,1]; title="state")
    is = eachindex(ξ.x)
    xs = [ξ.x(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, xs[:,i]), is)
    
    # ax = Axis(fig[1:2,2]; title="population")
    # ps = ([I(11) I(11)] * (xs.^2)')'
    # foreach(i->lines!(ax, ts, ps[:,i]), 1:11)


    ax = Axis(fig[1:2,2]; title="inputs")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)

    return fig
end
##


# ----------------------------------- tests ----------------------------------- ##


x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = [0.0]

smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
μ = @closure t->u0*sin(t)
α = @closure t->smooth(t, x0, xf, tf)
φ = PRONTO.Trajectory(θ,α,μ);

# μ = @closure t->SizedVector{1}(u0)
# φ = open_loop(θ,xf,μ,τ) # guess trajectory
ξ = pronto(θ,x0,φ,τ) # optimal trajectory

plot_spin(ξ,τ)
