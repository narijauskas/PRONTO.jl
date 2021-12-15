using PRONTO
using ForwardDiff
using ForwardDiff: derivative
using DifferentialEquations
using MatrixEquations
using LinearAlgebra
using GLMakie
using Interpolations



## ------------------------------ pre-optimization work ------------------------------ ## 


## ---------------- time ---------------- ##
T = 10.0 # final time
dt = 0.01
T0 = 0:dt:T

## ---------------- desired trajectory ---------------- ##
ϕ₀ = t -> tanh(t-5)*(1-tanh(t-5)^2)
amp = (π/2)*(1/maximum(ϕ₀.(T0)))

ϕ = t -> amp*tanh(t-5)*(1-tanh(t-5)^2)
dϕ = t -> derivative(ϕ, t)
ddϕ = t -> derivative(dϕ, t)
# sanity check:
# dϕ = t -> amp*(3tanh(t-5)^4 - 4tanh(t-5)^2 + 1)
# lines(T0, dϕ.(T0))

g = 9.8; l = 0.5;
u = t -> [(g*sin(ϕ(t)) - l*ddϕ(t)) / cos(ϕ(t))] # blows up
u = t -> [0]

x = t->[ϕ(t),dϕ(t)]
ξd = Trajectory(x,u)

# plot trajectory
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, T0, [ξd.x(t)[1] for t in T0])
lines!(ax, T0, [ξd.x(t)[2] for t in T0])
display(fig)

# regulator parameters
Qr = I(2)
Rr = I(1)

# cost parameters
Qc = I(2)
Rc = I(1)

# define dynamics
function ipend(x, u)
    # parameters:
    g = 9.8; l = 0.5;
    # dynamics:
    [x[2], (g/l)*sin(x[1]) - (u[1]/l) * cos(x[1])]
end

## linearize around desired trajectory
A = Jx(ipend, ξd)
B = Ju(ipend, ξd)

# create cost functional around linearized ξd
P₁,_ = arec(A(T), B(T)inv(Rc)B(T)', Qc) # solve algebraic riccati eq at time T
lc, mc = build_LQ_cost(ξd, Qc, Rc, P₁, T) # cost functional

# for LQR
Pr₁,_ = arec(A(T), B(T)inv(Rr)B(T)', Qr) # solve algebraic riccati eq at time T

P = optKr(ipend, ξd, Qr, Rr, Pr₁, T)

Kr = optKr(ipend, ξd, Qr, Rr, Pr₁, T)
Kritp = LinearInterpolation(T0, [Kr(t) for t in T0]) # interpolate to simplify type

plot(T0, [Kr(t)[1] for t in T0])
plot!(T0, [Kr(t)[2] for t in T0])

##




# p = (ipend, ξd, Kr, lc)
# prob = ODEProblem(ẋl!, ξd.x(0), (0.0,T), p) # IC syntax?
# solve(prob) # output syntax?
x = project(ξd, ipend, Kr, T)
u = PRONTO.project_u(ξd, x, Kr)
ξ = Trajectory(x, u)
# x_itp = LinearInterpolation(T0, [x(t) for t in T0]) # interpolate to simplify type
# u_itp = LinearInterpolation(T0, [u(t) for t in T0]) # interpolate to simplify type
# ξ = Trajectory(x_itp, u_itp)

##
fig = Figure()

lines(fig[1, 1], T0, [u(i)[1] for i in T0])
lines(fig[2, 1], T0, [ξ.x(i)[1] for i in T0])
fig

idx = findmax([u(i)[1] for i in T0])[2]
tidx = T0[idx]
Kr(tidx)
ξd.u(tidx)
findmax([ξd.x(t) for t in T0])

## ------------------------------ PRONTO loop ------------------------------ ## 
# inputs:
f = ipend
m = mc
l = lc

# linearize
h = ξ -> PRONTO.build_h(l, m, ξ, T)

Kᵣ = optKr(f, ξ, Qr, Rr, Pr₁, T) # ?
ξ = project(ξd, f, Kᵣ, l, T)
γ = 1
ζ = search_direction(ipend, ξ, ξd, Qc, Rc, P₁, Kr, T);
while γ > 0 # if keep γ as only condition, move initialization into loop?
    #TODO: is there a better way to check for convergence?
    ζ = search_direction()
    a, b, r1 = PRONTO.loss_grads1(l, m, ξ, T)
    Dh = build_Dh(a, b, r1) # returns Dh(ζ)
    # check for convergence here
    γ = armijo_backstep(ξ, ζ, f, Kᵣ, (h, l, Dh))
    ξ = ξ + γ*ζ
    Kᵣ = optKr(f, ξ, Q, R, P₁, T)
    ξ, lxi = project(ξ, f, Kᵣ, l, T) # update trajectory
    #MAYBE: print cost of that step (lxi)
end
