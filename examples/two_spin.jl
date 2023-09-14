using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct TwoSpin <: Model{4,1}
    kl::Float64 = 0.01
end

@define_f TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@define_l TwoSpin begin
    1/2*u'*kl*u
end

@define_m TwoSpin begin
    Pl = [1 0 0 0;0 0 0 0;0 0 1 0;0 0 0 0]
    1/2*x'*Pl*x
end

@define_Qr TwoSpin I(4)
@define_Rr TwoSpin I(1)
PRONTO.Pf(θ::TwoSpin, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))

resolve_model(TwoSpin)

## ----------------------------------- solve the problem ----------------------------------- ##

θ = TwoSpin() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [1.0, 0.0, 0.0, 0.0] # initial state
xf = @SVector [0.0, 1.0, 0.0, 0.0] # final state
μ = t->SVector{1}(0.4*sin(t)) # open loop input μ(t)
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory