

## ----------------------------------- dependencies ----------------------------------- ##


using PRONTO
using StaticArrays
using LinearAlgebra



NX = 4
NU = 1
NΘ = 0
struct TwoSpin <: PRONTO.Model{4,1,2}
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


# function PRONTO.preview(M::TwoSpin, ξ)
# end

PRONTO.generate_model(TwoSpin, dynamics, stagecost, termcost, Qreg, Rreg)
PRONTO.Pf(α,μ,tf,θ::TwoSpin) = SMatrix{4,4,Float64}(I(4))

# ----------------------------------- tests ----------------------------------- ##

θ = TwoSpin(1,1)
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = [0.1]
μ = @closure t->SizedVector{1}(u0)
φ = open_loop(θ,xf,μ,τ)
ξ = pronto(θ,x0,φ,τ)

##
# φ = PRONTO.guess_zi(M,θ,xf,u0,t0,tf)
# @time ξ = pronto(M,θ,t0,tf,x0,u0,φ)


# using GLMakie
# fig = Figure(); ax = Axis(fig[1,1])
# ts = LinRange(t0,tf,10001)
# is = eachindex(ξ.x)
# xs = [ξ.x(t)[i] for t∈ts, i∈is]
# foreach(i->lines!(ax, ts, xs[:,i]), is)
# display(fig)