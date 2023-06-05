using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

NX = 12
NU = 1

@kwdef struct xgate3 <: PRONTO.Model{NX,NU}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end

@dynamics xgate3 begin
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

@stage_cost xgate3 begin
    θ.kl/2*u'*I*u + 0.3*x'*mprod(diagm([0,0,1,0,0,1]))*x
end

@terminal_cost xgate3 begin
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(12)*(x-xf)
end

@regulatorQ xgate3 θ.kq*I(NX)
@regulatorR xgate3 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(xgate3)

# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::xgate3) = SMatrix{12,12,Float64}(I(12))


## ----------------------------------- run optimization ----------------------------------- ##

θ = xgate3()
τ = t0,tf = 0,10

ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ, x0, μ, τ) # guess trajectory
@time ξ = pronto(θ, x0, φ, τ) # optimal trajectory

##
import Pkg: activate
activate()
using MAT
activate(".")

ts = t0:0.01:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_xgate_10_1.0.mat","w")
write(file,"Uopt",us)
close(file)