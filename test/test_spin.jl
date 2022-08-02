using Revise
using BenchmarkTools, JET
using PRONTO
using LinearAlgebra
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
    maxiters = 10, #TODO: remove
)

# params = (
#     tol = 1e-5,
#     maxiters = 10,
# )

H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]


# model dynamics
f = @closure (x,u) -> collect((H0 + u[1]*H1)*x)


# stage cost
Ql = zeros(NX,NX)
Rl = 0.01
Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1] #terminal cost matrix Pl

l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
p = (x) -> 1/2*collect(x)'*Pl*collect(x)

@info "running autodiff"
model = autodiff(model,f,l,p)
# model = PRONTO.Model(NX,NU,f,l,p) # returns a Model{NX,NU}
@info "autodiff complete"




model = merge(model, (
    Qr = let M = Diagonal(SMatrix{NX,NX}(diagm([1,1,1,1])))
        (t)->M
    end, # can Qr be a function of α?
    Rr = let M = Diagonal(SMatrix{NU,NU}(diagm([1])))
        (t)->M
    end,
    iRr = let M = Diagonal(SMatrix{NU,NU}(inv(diagm([1]))))
        (t)->M
    end
))

##

# tx = @elapsed begin 
(η,stats) = pronto(model); PRONTO.overview(stats)
# end
# PRONTO.overview(stats)

#before: 16s
# @elapsed pronto(model)
# tx = map(1:10) do i
#     @elapsed pronto(model)
# end

##

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