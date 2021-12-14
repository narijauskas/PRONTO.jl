using PRONTO
using LinearAlgebra
using GLMakie

## ------------------------------ USER INPUTS ------------------------------ ## 
# desired trajectory


# regulator parameters
Qr = I(2)
Rr = I(1)

# cost parameters
Qc = I(2)
Rc = I(1)

# define dynamics
function fxn(x, u)
    # parameters:
    l = 1; g = 9.8;
    # dynamics:
    [x[2], (g/l)*sin(x[1]) - (u/l) * cos(x[1])]
end

T = 10 # final time

A = Jx(f, ξd)
B = Ju(f, ξd)

# create cost functional
P₁,_ = arec(A(T), B(T)inv(Rᵣ)B(T)', Qc) # solve algebraic riccati eq at time T
l, m = build_LQ_cost(ξd, Qc, Rc, P₁, T) # cost functional


Pr₁,_ = arec(A(T), B(T)inv(Rr)B(T)', Qr) # solve algebraic riccati eq at time T


# ---------------- from newt_invpend ---------------- #
dt = 0.01
T0 = 0:dt:10

ϕ₀ = t -> tanh(t-5)*(1-tanh(t-5)^2)
# lines(T0, ϕ₀.(T0))
amp = (π/2)*(1/maximum(ϕ₀.(T0)))

ϕ = t -> amp*tanh(t-5)*(1-tanh(t-5)^2)
dϕ = t -> derivative(ϕ, t)
ddϕ = t -> derivative(dϕ, t)
# sanity check:
# dϕ = t -> amp*(3tanh(t-5)^4 - 4tanh(t-5)^2 + 1)
# lines(T0, dϕ.(T0))

GG = 9.81;  LL = 0.5;
u = t -> (GG*sin(ϕ(t)) - LL*ddϕ(t))/cos(ϕ(t))
# u = t -> 0.0


## ------------------------------ DO PRONTO STUFF ------------------------------ ## 

