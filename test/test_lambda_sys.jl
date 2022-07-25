# using Test
using Revise
using PRONTO
using LinearAlgebra
using BenchmarkTools
# using WindowFunctions # blackman
using DSP.Windows: blackman
# https://docs.juliadsp.org/stable/windows/

NX = 6
NU = 4
ts = 0:0.001:5
model = (
    ts = ts,
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

# initial input

# function u_guess(t)
#     u = zeros(NU)
#     if t>2
#         u[1] = 5*blackman(601)
#     end
#     if t<3
#         u[3] = 5*blackman(601)
#     end
#     return u
# end

# μ = Interpolant(t->u_guess(t), ts, NU)

uguess = zeros(NU, length(ts))
uguess[1, 2001:end] = 5*blackman(3001)
uguess[3, 1:3001] = 5*blackman(3001)

μ = Interpolant(uguess, ts, NU)

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

ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU
α_ode = solve(ODEProblem(PRONTO.ol_dynamics!, model.x0, (0,T), (model.f, μ)))
α = Interpolant(t->α_ode(t), ts, NX)

##
(α,μ) = pronto(μ, model)
# NS = NX/2
I2 = collect([I(Int(NX/2)) I(Int(NX/2))])
pop(t) = I2*α(t).^2

using GLMakie

fig = Figure()
ax = Axis(fig[1,1])
for i in 1:Int(NX/2)
    lines!(ax, ts, map(t->pop(t)[i], ts))
end
ax = Axis(fig[2,1])
for i in 1:NX
    lines!(ax, ts, map(t->α_ode(t)[i], ts))
end
ax = Axis(fig[3,1])
for i in 1:NU
    lines!(ax, ts, map(t->μ(t)[i], ts))
end
display(fig)

