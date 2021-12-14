using PRONTO
using ForwardDiff
using ForwardDiff: derivative
using DifferentialEquations
using MatrixEquations
using LinearAlgebra
using GLMakie

## ------------------------------ USER INPUTS ------------------------------ ## 


# ---------------- time ---------------- #
T = 10 # final time
# unlike the og version, dt and T0 don't really matter beyond plotting
dt = 0.01
T0 = 0:dt:T

# ---------------- desired trajectory ---------------- #
ϕ₀ = t -> tanh(t-5)*(1-tanh(t-5)^2)
amp = (π/2)*(1/maximum(ϕ₀.(T0)))

ϕ = t -> amp*tanh(t-5)*(1-tanh(t-5)^2)
dϕ = t -> derivative(ϕ, t)
ddϕ = t -> derivative(dϕ, t)
# sanity check:
# dϕ = t -> amp*(3tanh(t-5)^4 - 4tanh(t-5)^2 + 1)
# lines(T0, dϕ.(T0))

g = 9.8; l = 0.5;
u = t -> [(g*sin(ϕ(t)) - l*ddϕ(t)) / cos(ϕ(t))]

x = t->[ϕ(t),dϕ(t)]
ξd = Trajectory(x,u)




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



# linearize around desired trajectory
A = Jx(ipend, ξd)
B = Ju(ipend, ξd)


# create cost functional
P₁,_ = arec(A(T), B(T)inv(Rc)B(T)', Qc) # solve algebraic riccati eq at time T
l, m = build_LQ_cost(ξd, Qc, Rc, P₁, T) # cost functional


Pr₁,_ = arec(A(T), B(T)inv(Rr)B(T)', Qr) # solve algebraic riccati eq at time T






## ------------------------------ DO PRONTO STUFF ------------------------------ ## 

