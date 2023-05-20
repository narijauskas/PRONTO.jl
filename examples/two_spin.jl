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

## ----------------------------------- generate solver kernel ----------------------------------- ##

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


## ----------------------------------- run optimization ----------------------------------- ##

θ = TwoSpin()
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = 0.1
μ = @closure t->SizedVector{1}(u0)
φ = open_loop(θ, xf, μ, τ) # guess trajectory
ξ = pronto(θ, x0, φ, τ) # optimal trajectory
@time ξ = pronto(θ, x0, φ, τ) # optimal trajectory
# @code_warntype PRONTO.f(x0,u0,t0,θ)




## ----------------------------------- symbolic ----------------------------------- ##

θ = symbolic(TwoSpin)
λ = symbolic(:λ, 1:NX)
x = symbolic(:x, 1:NX)
u = symbolic(:u, 1:NU)
t = symbolic(:t)

PRONTO.f(x,u,t,θ)
PRONTO.fx(x,u,t,θ)
PRONTO.fu(x,u,t,θ)

PRONTO.l(x,u,t,θ)
PRONTO.lx(x,u,t,θ)
PRONTO.lu(x,u,t,θ)

PRONTO.Q(x,u,t,θ)

PRONTO.Lxx(λ,x,u,t,θ)
PRONTO.Lxu(λ,x,u,t,θ)
PRONTO.Luu(λ,x,u,t,θ)