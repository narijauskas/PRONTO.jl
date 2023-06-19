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

# get the ith eigenstate
function x_eig(i)
    n = 4
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr == θ[2] and θ.kq == θ[3]


# ------------------------------- split system to eigenstate 2 ------------------------------- ##

@kwdef struct Split1 <: Model{18,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost1(x,u,t,θ)
    P = I(18) - inprod(x_eig(4))
    1/2 * collect(x')*P*x
end


# ------------------------------- split system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 4
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    return mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

stagecost(x,u,t,θ) = 1/2 *θ[1]*collect(u')I*u


regR(x,u,t,θ) = θ.kr*I(1)
function regQ(x,u,t,θ)
    x_re = x[1:9]
    x_im = x[10:18]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(9) - ψ*ψ')
end



# PRONTO.Pf(α,μ,tf,θ::Split2) = SMatrix{22,22,Float64}(I(22) - α*α')
PRONTO.Pf(α,μ,tf,θ::Split1) = SMatrix{18,18,Float64}(I(18) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

# PRONTO.generate_model(Split2, dynamics, stagecost, termcost2, regQ, regR)
PRONTO.generate_model(Split1, dynamics, stagecost, termcost1, regQ, regR)


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

    ax = Axis(fig[1:2,2]; title="fidelity")
    fs = [ξ.x(t)'inprod(x_eig(i))*ξ.x(t) for t∈ts,i∈1:5]
    foreach(i->lines!(ax, ts, fs[:,i]), 1:5)

    ax = Axis(fig[3,1:2]; title="inputs")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)

    return fig
end


## ------------------------------- demo: eigenstate 1->2 in 10s ------------------------------- ##


# x0 = SVector{22}(x_eig(1))
# xf = SVector{22}(x_eig(2))
# t0,tf = τ = (0,10)


# θ = Split2(kl=0.01, kr=1, kq=1)
# μ = @closure t->SVector{1}(0.4*sin(t))
# φ = open_loop(θ,x0,μ,τ)
# @time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, limitγ = true)

# plot_split(ξ,τ)


## ------------------------------- demo: eigenstate 1->4 in 10s ------------------------------- ##

x0 = SVector{18}(x_eig(1))
# xf = SVector{22}((x_eig(4) + x_eig(5))/sqrt(2))
t0,tf = τ = (0,1.5)


θ = Split1(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 100, limitγ = true)

# plot_split(ξ,τ)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_nu4_1.5T.mat", "w")
write(file, "Uopt", us)
close(file)


## ------------------------------- demo: eigenstate 1->4 in 2s ------------------------------- ##

x0 = SVector{22}(x_eig(1))
xf = SVector{22}(x_eig(4))
t0,tf = τ = (0,2.55)


θ = Split4(kl=0.02, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 100, limitγ = true)

plot_split(ξ,τ)








## ------------------------------- step-by-step debugging ------------------------------- ##

Kr = PRONTO.regulator(θ,φ,τ)
ξ = PRONTO.projection(θ,x0,φ,Kr,τ)

λ = PRONTO.lagrangian(θ,ξ,φ,Kr,τ)
Ko = PRONTO.optimizer(θ,λ,ξ,φ,τ)
vo = PRONTO.costate(θ,λ,ξ,φ,Ko,τ)


ζ = PRONTO.search_direction(θ,ξ,Ko,vo,τ)
γ = 0.7
ξ1 = PRONTO.armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)


## ------------------------------- other ------------------------------- ##


# shorter error messages:
using SciMLBase, DifferentialEquations
Base.show(io::IO, ::Type{<:SciMLBase.ODEProblem}) = print(io, "ODEProblem{...}")
Base.show(io::IO, ::Type{<:OrdinaryDiffEq.ODEIntegrator}) = print(io, "ODEIntegrator{...}")


