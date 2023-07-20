using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef


@kwdef struct TwoSpin <: Model{4,1}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


## --------------------- option 2 --------------------- ##

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

PRONTO.Pf(θ::TwoSpin, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))

# must be run after any changes to model definition
resolve_model(TwoSpin)


PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))

## --------------------- run optimization --------------------- ##

θ = TwoSpin() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [0.0, 1.0, 0.0, 0.0] # initial state
xf = @SVector [1.0, 0.0, 0.0, 0.0] # final state
μ = t->[0.1] # open loop input μ(t)
η = open_loop(θ, xf, μ, τ); # guess trajectory
η0 = open_loop(θ, x0, μ, τ); # guess trajectory
ξ,data = pronto(θ, x0, η0, τ); # optimal trajectory
@time ξ,data = pronto(θ, x0, η0, τ); # optimal trajectory
