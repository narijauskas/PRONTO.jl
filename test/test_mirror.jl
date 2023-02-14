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
    n = 2
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    # x_eig = kron([1;0],w[:,i])
    x_eig = w[:,i]
end

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr == θ[2] and θ.kq == θ[3]


# ------------------------------- mirror system on eigenstate 4 ------------------------------- ##

@kwdef struct Reflect4 <: Model{25,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    xf = vec(I(5) - 2*x_eig(4)*x_eig(4)')
    P = I(25)
    1/2 * collect((x-xf)')*P*(x-xf)
end


# ------------------------------- reflect system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 2
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H00 = kron(I(5),H0)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))   
    H11 = kron(I(5),H1)
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    H22 = kron(I(5),H2)
    return -im*ω*(H00 + sin(u[1])*H11 + (1-cos(u[1]))*H22) *x
end

stagecost(x,u,t,θ) = 1/2 *θ.kl*collect(u')I*u

regR(x,u,t,θ) = θ.kr*I(1)
function regQ(x,u,t,θ)
    θ.kq*I(25)
end


PRONTO.Pf(α,μ,tf,θ::Reflect4) = SMatrix{25,25,Float64}(I(25))

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(Reflect4, dynamics, stagecost, termcost, regQ, regR)



## ------------------------------- plots ------------------------------- ##

import Pkg: activate
activate()
using GLMakie, Statistics
activate(".")
include("../dev/plot_setup.jl")
# plot_split(ξ,τ)

function plot_reflect(ξ,τ)
    fig = Figure()
    ts = LinRange(τ...,10001)

    # ax = Axis(fig[1:2,1]; title="state")
    # is = eachindex(ξ.x)
    # xs = [ξ.x(t)[i] for t∈ts, i∈is]
    # foreach(i->lines!(ax, ts, xs[:,i]), is)
    
    ax = Axis(fig[1:2,1]; title="population")
    is = eachindex(ξ.x)
    xs = [ξ.x(t)[i] for t∈ts, i∈is]
    ps = ([I(22) I(22)] * (xs.^2)')'
    foreach(i->lines!(ax, ts, ps[:,i]), 1:11)

    # ax = Axis(fig[1:2,2]; title="fidelity")
    # fs = [ξ.x(t)'inprod(x_eig(i))*ξ.x(t) for t∈ts,i∈1:4]
    # foreach(i->lines!(ax, ts, fs[:,i]), 1:4)

    ax = Axis(fig[1:2,2]; title="inputs")
    is = eachindex(ξ.u)
    us = [ξ.u(t)[i] for t∈ts, i∈is]
    foreach(i->lines!(ax, ts, us[:,i]), is)

    return fig
end


## ------------------------------- demo: eigenstate 4->4 in 4 ------------------------------- ##

ψ0 = zeros(11,1)
ψ0[1:5] = x_eig(4)[1:5]
ψf = zeros(11,1)
ψf[7:11] = x_eig(4)[7:11]
x0 = SVector{44}(vec([ψ0;ψf;0*ψ0;0*ψf]))
t0,tf = τ = (0,2.8)


θ = Reflect4(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(1.0*sin(12*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 100, limitγ = true)

plot_reflect(ξ,τ)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_4hk_mirror.mat", "w")
write(file, "Uopt", us)
close(file)
