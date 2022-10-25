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









## ----------------------------------- dependencies ----------------------------------- ##


using PRONTO
using StaticArrays
using LinearAlgebra



NX = 4
NU = 1
NΘ = 0
struct TwoSpin <: PRONTO.Model{NX,NU,NΘ}
end

# ----------------------------------- model definition ----------------------------------- ##

let
    # model dynamics
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    f = (θ,t,x,u) -> collect((H0 + u[1]*H1)*x)


    # stage cost
    Ql = zeros(NX,NX)
    Rl = 0.01
    l = (θ,t,x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)

    # terminal cost
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    p = (θ,t,x,u) -> 1/2*collect(x)'*Pl*collect(x)

    # regulator
    Rr = (θ,t,x,u) -> diagm([1])
    Qr = (θ,t,x,u) -> diagm([1,1,1,1])
    # Pr(θ,t,x,u)

    @derive TwoSpin
end


## ----------------------------------- tests ----------------------------------- ##

t0 = 0.0
tf = 10.0
x0 = [0.0;1.0;0.0;0.0]
xf = [1.0;0.0;0.0;0.0]
u0 = [0.0]


M = TwoSpin()
x = x0
u = [0.0]
t = t0
θ = nothing
P = collect(I(NX))
buf = similar(x)

PRONTO.f(M,θ,t,x,u)
PRONTO.fx(M,θ,t,x,u)
PRONTO.Rr(M,θ,t,x,u)
PRONTO.Kr(M,θ,t,x,u,P)
PRONTO.f!(M,buf,θ,t,x,u)

##
φg = PRONTO.guess_zi(M,θ,x0,u0,t0,tf)
pronto(M,θ,t0,tf,x0,u0,φg)