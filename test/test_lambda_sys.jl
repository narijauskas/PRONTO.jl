# using Test
using Revise
using PRONTO
using LinearAlgebra
using BenchmarkTools


NX = 6
NU = 4
model = (
    ts = 0:0.001:10,
    NX = NX,
    NU = NU,
    x0 = [1.0;0.0;0.0;0.0;0.0;0.0],
    tol = 1e-4,
    # x_eq = zeros(NX),
    x_eq = [0.0;0.0;1.0;0.0;0.0;0.0], #For this example, we don't have any equilibrium points... xf is the target state here 
    u_eq = zeros(NU),
    maxiters = 20,
    α = 0.4,
    β = 0.7,
)


# model parameters
H0 = [0 0 0 -0.5 0 0;0 0 0 0 0 0;0 0 0 0 0 -0.5;0.5 0 0 0 0 0;0 0 0 0 0 0;0 0 0.5 0 0 0]
H1 = [0 0 0 0 -0.5 0;0 0 0 -0.5 0 0;0 0 0 0 0 0;0 0.5 0 0 0 0;0.5 0 0 0 0 0;0 0 0 0 0 0]
H2 = [0 -0.5 0 0 0 0;0.5 0 0 0 0 0;0 0 0 0 0 0;0 0 0 0 -0.5 0;0 0 0 0.5 0 0;0 0 0 0 0 0]
H3 = [0 0 0 0 0 0;0 0 0 0 0 -0.5;0 0 0 0 -0.5 0;0 0 0 0 0 0;0 0 0.5 0 0 0;0 0.5 0 0 0 0]
H4 = [0 0 0 0 0 0;0 0 -0.5 0 0 0;0 0.5 0 0 0 0;0 0 0 0 0 0;0 0 0 0 0 -0.5;0 0 0 0 0.5 0]


# model dynamics
function f(x,u)
    
    f = collect((H0 + u[1]*H1 + u[2]*H2 + u[3]*H3 + u[4]*H4)*x)
    return f

end

# stage cost
Ql = zeros(NX,NX)
Rl = 0.01
Pl = [1 0 0 0 0 0;0 1 0 0 0 0;0 0 0 0 0 0;0 0 0 1 0 0;0 0 0 0 1 0;0 0 0 0 0 0] #terminal cost matrix Pl

l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
p = (x)-> 1/2*collect(x)'*Pl*collect(x)

@info "running autodiff"
model = autodiff(model,f,l,p)
@info "autodiff complete"

model = merge(model, (
    Qr = Interpolant(t->1.0*diagm([1,1,1,1,1,1]), model.ts, NX, NX), #TODO: can Qr be a function of α?
    Rr = Interpolant(t->1.0*diagm([1,1,1,1]), model.ts, NU, NU),
    iRr = Interpolant(t->inv(1.0*diagm([1,1,1,1])), model.ts, NU, NU),
))

(α,μ) = pronto(model)
# NS = NX/2
I2 = collect([I(3) I(3)])
pop(t) = I2*α(t).^2

using GLMakie

ts = model.ts
fig = Figure()
ax = Axis(fig[1,1])
for i in 1:3
    lines!(ax, ts, map(t->pop(t)[i], ts))
end
ax = Axis(fig[2,1])
for i in 1:NU
    lines!(ax, ts, map(t->μ(t)[i], ts))
end
display(fig)

