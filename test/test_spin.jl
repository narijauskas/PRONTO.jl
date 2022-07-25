using Revise
using PRONTO
using LinearAlgebra

NX = 4
NU = 1
model = (
    ts = 0:0.001:10,
    NX = NX,
    NU = NU,
    x0 = [0.0;1.0;0.0;0.0],
    tol = 1e-5,
    # x_eq = zeros(NX),
    x_eq = [1.0;0.0;0.0;0.0], #For this example, we don't have any equilibrium points... xf is the target state here 
    u_eq = zeros(NU),
    maxiters = 20,
    α = 0.4,
    β = 0.7,
)


H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]


# model dynamics
function f(x,u)
    f = collect((H0 + u[1]*H1)*x)
    return f
end

# stage cost
Ql = zeros(NX,NX)
Rl = 0.01
Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1] #terminal cost matrix Pl

l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
p = (x)-> 1/2*collect(x)'*Pl*collect(x)

@info "running autodiff"
model = autodiff(model,f,l,p)
@info "autodiff complete"

model = merge(model, (
    Qr = Interpolant(t->1.0*diagm([1,1,1,1]), model.ts, NX, NX), #TODO: can Qr be a function of α?
    Rr = Interpolant(t->1.0*diagm([1]), model.ts, NU, NU),
    iRr = Interpolant(t->inv(1.0*diagm([1])), model.ts, NU, NU),
))

η = pronto(model)
ts = model.ts



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