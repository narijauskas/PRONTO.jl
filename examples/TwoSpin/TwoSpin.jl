using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

NX = 4
NU = 1
NΘ = 2

@kwdef struct TwoSpin{T} <: PRONTO.Model{NX,NU,NΘ}
    kr::T = 1.0
    kq::T = 1.0
end


@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@stage_cost TwoSpin begin
    Rl = [0.01;;]
    1/2 * u'*Rl*u
end

@terminal_cost TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*x'*Pl*x
end

@regulatorQ TwoSpin θ.kq*I(NX)
@regulatorR TwoSpin θ.kr*I(NU)
@lagrangian TwoSpin

# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::TwoSpin{T}) where T = SMatrix{4,4,T}(I(4))

info(PRONTO.as_bold("TwoSpin")*" model ready")