using PRONTO
using StaticArrays, LinearAlgebra

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end


function inprod(x)
    i = Int(length(x)/2)
    a = x[1:i]
    b = x[i+1:end]
    P = [a*a'+b*b' -(a*b'+b*a');
        a*b'+b*a' a*a'+b*b']
    return P
end


function x_eig(i)
    H0 = [1 0;0 -1]
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- 2spin system to eigenstate 2 ------------------------------- ##

@kwdef struct spin2 <: Model{4,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    P = I(4) - inprod(x_eig(2))
    1/2 * collect(x')*P*x
end


# ------------------------------- 2spin system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    H0 = [1 0;0 -1]
    H1 = [0 -1im;1im 0]
    return mprod(-im*(H0+u[1]*H1))*x
end


stagecost(x,u,t,θ) = θ.kl*collect(u')*I*u

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:2]
    x_im = x[3:4]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(2) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::spin2) = SMatrix{4,4,Float64}(I(4) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(spin2, dynamics, stagecost, termcost, regQ, regR)


## ------------------------------- plots ------------------------------- ##

import Pkg: activate
activate()
using GLMakie, Statistics
activate(".")
include("../dev/plot_setup.jl")
# plot_split(ξ,τ)

function plot_spin2(ξ,τ)
    fig = Figure()
    ts = LinRange(τ...,1001)

    ax = Axis(fig[1:2,1]; title="input")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)
    
    ax = Axis(fig[1:2,2]; title="population")
    is = eachindex(ξ.x)
    xs = [ξ.x(t)[i] for t∈ts, i∈is]
    ps = ([I(2) I(2)] * (xs.^2)')'
    foreach(i->lines!(ax, ts, ps[:,i]), 1:2)

    return fig
end


## ------------------------------- demo: eigenstate 1->2 in 10 ------------------------------- ##

x0 = SVector{4}(x_eig(1))

θ = spin2(kl=0.01, kr=1, kq=1)

t0,tf = τ = (0,10)

μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

plot_spin2(ξ,τ)
