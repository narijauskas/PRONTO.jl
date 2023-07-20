using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef


## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct TwoSpin <: Model{4,1}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end

@define_f TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@define_l TwoSpin begin
    Rl = [0.01;;]
    1/2*u'*Rl*u
end

@define_m TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*x'*Pl*x
end

@define_Q TwoSpin kq*I(4)
@define_R TwoSpin kr*I(1)
resolve_model(TwoSpin)


# show the population function on each iteration
PRONTO.preview(θ::TwoSpin, ξ) = [I(2) I(2)]*(ξ.x.^2)
PRONTO.γmax(θ::TwoSpin, ζ, τ) = PRONTO.sphere(1, ζ, τ)
PRONTO.Pf(θ::TwoSpin, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))

## ----------------------------------- solve the problem ----------------------------------- ##

θ = TwoSpin() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [0.0, 1.0, 0.0, 0.0] # initial state
xf = @SVector [1.0, 0.0, 0.0, 0.0] # final state
μ = t->[0.1] # open loop input μ(t)
η = open_loop(θ, xf, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ; show_armijo=true, show_info=false, show_preview=false); # optimal trajectory
