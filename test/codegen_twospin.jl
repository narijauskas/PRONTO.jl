using Test, PRONTO

using StaticArrays
using LinearAlgebra
using Base: @kwdef

NX = 4
NU = 1
NΘ = 2

@kwdef struct TwoSpin{T} <: PRONTO.Model{NX,NU,NΘ}
    kr::T = 1
    kq::T = 1
end

## ----------------------------------- build ----------------------------------- ##

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

## ----------------------------------- test ----------------------------------- ##






θ = symbolic(TwoSpin)
x = symbolic(:x, 1:NX)
u = symbolic(:u, 1:NU)
t = symbolic(:t)

PRONTO.f(x,u,t,θ)
PRONTO.fx(x,u,t,θ)
PRONTO.fu(x,u,t,θ)


# symbolic(:x, 1:NX)

θ0 = TwoSpin{Float64}() # make an instance of the mode.
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = 0.1

PRONTO.fx(x0,u0,t0,θ0)
