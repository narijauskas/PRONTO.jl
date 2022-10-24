using Revise
using BenchmarkTools, JET
using PRONTO
using LinearAlgebra
using SparseArrays
using StaticArrays
using FastClosures

NX = 4
NU = 1
T = 10
ts = 0:0.001:T
model = (
    NX = NX,
    NU = NU,
    T = T, #TODO: remove
    ts = ts, #TODO: remove
    x0 = [0.0;1.0;0.0;0.0],
    u0 = zeros(NU),
    xf = [1.0;0.0;0.0;0.0], #For this example, we don't have any equilibrium points... xf is the target state here
    uf = zeros(NU),
    tol = 1e-5, #TODO: remove
    maxiters = 15, #TODO: remove
)


##
(η,stats) = pronto(model); PRONTO.overview(stats)
##

# pronto(model,reg,t,x0)

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









## ----------------------------------- kernel definition ----------------------------------- ##


using PRONTO
using StaticArrays
using LinearAlgebra



NX = 4
NU = 1
NΘ = 0
struct TwoSpin <: PRONTO.Model{NX,NU,NΘ}
end

##
let
    # model dynamics
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    f = (x,u,t,θ) -> collect((H0 + u[1]*H1)*x)


    # stage cost
    Ql = zeros(NX,NX)
    Rl = 0.01
    l = (x,u,t,θ) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)

    # terminal cost
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    p = (x,u,t,θ) -> 1/2*collect(x)'*Pl*collect(x)

    # regulator
    Rr = (x,u,t,θ) -> diagm([1])
    Qr = (x,u,t,θ) -> diagm([1,1,1,1])
    # Pr(x,u,t,θ)

    @derive TwoSpin
end

t0 = 0.0
tf = 10.0
x0 = [0.0;1.0;0.0;0.0]
xf = [1.0;0.0;0.0;0.0]

PRONTO.Kr(M,x,u,t,θ,P)
## ----------------------------------- tests ----------------------------------- ##
M = TwoSpin()
x = x0
u = [0.0]
t = t0
θ = nothing
P = collect(I(NX))
buf = similar(x)

PRONTO.f(M,x,u,t,θ)
PRONTO.fx(M,x,u,t,θ)
PRONTO.Rr(M,x,u,t,θ)
PRONTO.Kr(M,x,u,t,θ,P)
PRONTO.f!(buf,M,x,u,t,θ)