using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define helper functions ----------------------------------- ##

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct XGate3 <: PRONTO.Model{12,1}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end

@define_f XGate3 begin
    E0 = 0.0
    E1 = 1.0
    E2 = 5.0
    H0 = diagm([E0, E1, E2])
    H00 = kron(I(2),H0)
    a1 = 0.1
    a2 = 0.5
    a3 = 0.3
    Ω1 = a1 * u[1]
    Ω2 = a2 * u[1]
    Ω3 = a3 * u[1]
    H1 = [0 Ω1 Ω3; Ω1 0 Ω2; Ω3 Ω2 0]
    H11 = kron(I(2),H1)
    return 2 * π * mprod(-im * (H00 + H11)) * x
end

@define_l XGate3 begin
    kl/2*u'*I*u + 0.3*x'*mprod(diagm([0,0,1,0,0,1]))*x
end

@define_m XGate3 begin
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(12)*(x-xf)
end

@define_Qr XGate3 kq*I(12)
@define_Rr XGate3 kr*I(1)

# must be run after any changes to model definition
resolve_model(XGate3)

# overwrite default behavior of Pf
PRONTO.Pf(θ::XGate3,α,μ,tf) = SMatrix{12,12,Float64}(I(12))
PRONTO.γmax(θ::XGate3, ζ, τ) = PRONTO.sphere(2, ζ, τ)
PRONTO.preview(θ::XGate3, ξ) = ξ.u

## ----------------------------------- run optimization ----------------------------------- ##

θ = XGate3()
τ = t0,tf = 0,10

ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ); # optimal trajectory

