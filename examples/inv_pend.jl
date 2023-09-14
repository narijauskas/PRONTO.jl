using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct InvPend <: Model{2,1}
    L::Float64 = 2 
    g::Float64 = 9.81 
    Q::SMatrix{2,2,Float64} = I(2) 
    R::SMatrix{1,1,Float64} = I(1) 
    P::SMatrix{2,2,Float64} = I(2)
end

@define_f InvPend [
    x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
]
@define_l InvPend 1/2*x'*Q*x + 1/2*u'*R*u
@define_m InvPend 1/2*x'*I(2)*x
@define_Qr InvPend diagm([10, 1])
@define_Rr InvPend diagm([1e-3])
resolve_model(InvPend)
PRONTO.preview(θ::InvPend, ξ) = ξ.x

## ----------------------------------- solve the problem ----------------------------------- ##

θ = InvPend() 
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]

α = t->xf
μ = t->u0
η = closed_loop(θ,x0,α,μ,τ)

ξ,data = pronto(θ,x0,η,τ; tol=1e-3);
