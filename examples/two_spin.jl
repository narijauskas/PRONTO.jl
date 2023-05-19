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


## ----------------------------------- model definition ----------------------------------- ##
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


@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end


@stage_cost TwoSpin begin
    Rl = [0.01;;]
    1/2 * collect(u')*Rl*u
end

@terminal_cost TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*collect(x')*Pl*x
end

@regulatorQ TwoSpin θ[1]*I(NU)
@regulatorR TwoSpin θ[2]*I(NX)

@lagrangian TwoSpin

PRONTO.generate_model(TwoSpin, dynamics, stagecost, termcost, Qreg, Rreg)
PRONTO.build_f(TwoSpin, dynamics)
PRONTO.build_l(TwoSpin, stagecost)
PRONTO.build_p(TwoSpin, termcost)






# PRONTO.build_f
# PRONTO.build_l
# PRONTO.build_L
# PRONTO.build_p
# PRONTO.build_QR

# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::TwoSpin) = SMatrix{4,4,Float64}(I(4))

## ----------------------------------- tests ----------------------------------- ##

θ = TwoSpin(1,1) # make an instance of the mode.
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = [0.1]
μ = @closure t->SizedVector{1}(u0)
φ = open_loop(θ, xf, μ, τ) # guess trajectory
ξ = pronto(θ, x0, φ, τ) # optimal trajectory
