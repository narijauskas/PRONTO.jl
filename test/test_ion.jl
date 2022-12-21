using PRONTO
using StaticArrays, LinearAlgebra


# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr == θ[2] and θ.kq == θ[3]


# ------------------------------- split system to eigenstate 2 ------------------------------- ##

@kwdef struct ion <: Model{4,2,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end

function termcost(x,u,t,θ)
    ω = 1.0
    m = 1.0
    x1 = 0.0
    x2 = 1.0
    # m/2 * ω^2 * (x[1]-x2)^2 + 1/2 * m * x[2]^2 + 1/2 * x[3]^2 + 1/2 * (x[4]-1)^2
    xf = [x2;0.0;0.0;1.0]
    1/2 * collect((x-xf)')*diagm([m*ω^2,m,1,1])*(x-xf)
end

# ------------------------------- split system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    m = 1.0
    x1 = 0.0
    x2 = 1.0
     [
        x[2],
        (-1/m)*(x[3]*m*ω^2*(x[1]-x1) + x[4]*m*ω^2*(x[1]-x2)),
        u[1],
        u[2]
    ]
end

stagecost(x,u,t,θ) = 1/2 *θ.kl*collect(u')*I(2)*u
regR(x,u,t,θ) = θ.kr*I(2)
regQ(x,u,t,θ) = θ.kq*I(4)


PRONTO.Pf(α,μ,tf,θ::ion) = SMatrix{4,4,Float64}(I(4))

# ------------------------------- generate model and derivatives ------------------------------- ##

# PRONTO.generate_model(Split2, dynamics, stagecost, termcost2, regQ, regR)
PRONTO.generate_model(ion, dynamics, stagecost, termcost, regQ, regR)



## ------------------------------- plots ------------------------------- ##

import Pkg: activate
activate()
using GLMakie, Statistics
activate(".")
include("../dev/plot_setup.jl")
# plot_split(ξ,τ)

function plot_split(ξ,τ)
    fig = Figure()
    ts = LinRange(τ...,10001)

    ax = Axis(fig[1:2,1]; title="state")
    is = eachindex(ξ.x)
    xs = [ξ.x(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, xs[:,i]), is)
    
    # ax = Axis(fig[1:2,2]; title="population")
    # ps = ([I(11) I(11)] * (xs.^2)')'
    # foreach(i->lines!(ax, ts, ps[:,i]), 1:11)

    # ax = Axis(fig[1:2,2]; title="fidelity")
    # fs = [ξ.x(t)'inprod(x_eig(i))*ξ.x(t) for t∈ts,i∈1:4]
    # foreach(i->lines!(ax, ts, fs[:,i]), 1:4)

    ax = Axis(fig[1:2,2]; title="inputs")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)

    return fig
end


## ------------------------------- demo: eigenstate 1->2 in 10s ------------------------------- ##


x0 = [0.0;0.0;1.0;0.0]
xf = [1.0;0.0;0.0;1.0]
t0,tf = τ = (0,10)


θ = ion(kl=0.01, kr=1, kq=1)
# μ = @closure t->SVector{2}([0.5*sin(t);0.5*sin(t)])
μ = @closure t->SVector{2}(zeros(2))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, limitγ = true)

plot_split(ξ,τ)

