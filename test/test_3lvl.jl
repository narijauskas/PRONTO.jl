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
    E0 = 0
    E1 = 1
    E2 = 5
    H0 = diagm([E0, E1, E2])
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- 3lvl system to eigenstate 2 ------------------------------- ##

@kwdef struct lvl3 <: Model{6,1,4}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
    T::Float64 # time horizon T
end


function termcost(x,u,t,θ)
    P = I(6) - inprod(x_eig(2))
    1/2 * 10 * collect(x')*P*x
end


# ------------------------------- 3lvl system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    E0 = 0
    E1 = 1
    E2 = 5
    H0 = diagm([E0, E1, E2])
    a1 = 0.1
    a2 = 0.5
    a3 = 0.3
    Ω1 = a1*u[1]
    Ω2 = a2*u[1]
    Ω3 = a3*u[1]
    H1 = [0 Ω1 Ω3;Ω1 0 Ω2;Ω3 Ω2 0]
    return 2*π*mprod(-im*(H0+H1))*x
end

# stagecost(x,u,t,θ) = 1/2 * (θ[1]*collect(u')I*u + 1*collect(x')*mprod(diagm([0, 0, 1]))*x)

# function kll(t)
#     return θ.kl+10*(exp(-2*t)+exp(2*t-2*θ.T))
# end

function kll(t)
    return max(θ.kl,100*10^(-2*t),100*10^(2*t-2*θ.T))
end

stagecost(x,u,t,θ) = 15/2*(kll(t)*collect(u')I*u + 1*collect(x')*mprod(diagm([0, 0, 1]))*x)/θ.T


regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:3]
    x_im = x[4:6]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(3) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::lvl3) = SMatrix{6,6,Float64}(I(6) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(lvl3, dynamics, stagecost, termcost, regQ, regR)


## ------------------------------- plots ------------------------------- ##

import Pkg: activate
activate()
using GLMakie, Statistics
activate(".")
include("../dev/plot_setup.jl")
# plot_split(ξ,τ)

function plot_3lvl(ξ,τ)
    fig = Figure()
    ts = LinRange(τ...,1001)

    ax = Axis(fig[1:2,1]; title="input")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)
    
    ax = Axis(fig[1:2,2]; title="population")
    is = eachindex(ξ.x)
    xs = [ξ.x(t)[i] for t∈ts, i∈is]
    ps = ([I(3) I(3)] * (xs.^2)')'
    foreach(i->lines!(ax, ts, ps[:,i]), 1:3)

    return fig
end


## ------------------------------- demo: eigenstate 0->1 in 10 ------------------------------- ##

x0 = SVector{6}(x_eig(1))

θ = lvl3(kl=0.01, kr=1, kq=1, T=10)

t0,tf = τ = (0,θ.T)

μ = @closure t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

plot_3lvl(ξ,τ)

## ------------------------------- demo: eigenstate 0->1 in 15 ------------------------------- ##

x0 = SVector{6}(x_eig(1))

θ = lvl3(kl=0.01, kr=1, kq=1, T=15)

t0,tf = τ = (0,θ.T)

μ = @closure t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

plot_3lvl(ξ,τ)

## ------------------------------- demo: eigenstate 0->1 in 20 ------------------------------- ##

x0 = SVector{6}(x_eig(1))

θ = lvl3(kl=0.01, kr=1, kq=1, T=20)

t0,tf = τ = (0,θ.T)

μ = @closure t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

plot_3lvl(ξ,τ)

## ------------------------------- demo: eigenstate 0->1 in 50 ------------------------------- ##

x0 = SVector{6}(x_eig(1))

θ = lvl3(kl=0.01, kr=1, kq=1, T=50)

t0,tf = τ = (0,θ.T)

μ = @closure t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

plot_3lvl(ξ,τ)

## ------------------------------- demo: eigenstate 0->1 in 100 ------------------------------- ##

x0 = SVector{6}(x_eig(1))

θ = lvl3(kl=0.01, kr=1, kq=1, T=100)

t0,tf = τ = (0,θ.T)

μ = @closure t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

plot_3lvl(ξ,τ)