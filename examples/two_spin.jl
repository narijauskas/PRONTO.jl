using PRONTO
using StaticArrays
using LinearAlgebra

NX = 4
NU = 1
NΘ = 2
struct TwoSpin <: PRONTO.Model{NX,NU,NΘ}
    kr::Float64
    kq::Float64
end


# ----------------------------------- model definition ----------------------------------- ##
function dynamics(x,u,t,θ)
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

Rreg(x,u,t,θ) = θ[1]*I(NU)
Qreg(x,u,t,θ) = θ[2]*I(NX)

function stagecost(x,u,t,θ)
    Rl = [0.01;;]
    1/2 * collect(u')*Rl*u
end

function termcost(x,u,t,θ)
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*collect(x')*Pl*x
end

PRONTO.generate_model(TwoSpin, dynamics, stagecost, termcost, Qreg, Rreg)

# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::TwoSpin) = SMatrix{4,4,Float64}(I(4))

# ----------------------------------- tests ----------------------------------- ##

θ = TwoSpin(1,1)
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = [0.0]

smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
μ = @closure t->u0*sin(t)
α = @closure t->smooth(t, x0, xf, tf)
φ = PRONTO.Trajectory(θ,α,μ);

# μ = @closure t->SizedVector{1}(u0)
# φ = open_loop(θ,xf,μ,τ) # guess trajectory
ξ = pronto(θ,x0,φ,τ) # optimal trajectory
