using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

NX = 4
NU = 1

@kwdef struct TwoSpin <: PRONTO.Model{NX,NU}
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

@define_Q TwoSpin θ.kq*I(NX)
@define_R TwoSpin θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(TwoSpin)


# PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x, (1,3)))
PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))


# overwrite default behavior of Pf for TwoSpin models
PRONTO.Pf(α,μ,tf,θ::TwoSpin) = SMatrix{4,4,Float64}(I(4))
# PRONTO.Pf(θ::TwoSpin,α,μ,tf) = SMatrix{4,4,Float64}(I(4))
# PRONTO.Pf(model::TwoSpin,α,μ,tf) = SMatrix{4,4,Float64}(I(4))

## --------------------- run optimization --------------------- ##

θ = TwoSpin() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [0.0, 1.0, 0.0, 0.0] # initial state
xf = @SVector [1.0, 0.0, 0.0, 0.0] # final state
μ = t->[0.1] # open loop input μ(t)
η = open_loop(θ, xf, μ, τ) # guess trajectory
η0 = open_loop(θ, x0, μ, τ) # guess trajectory
ξ = pronto(θ, x0, η0, τ) # optimal trajectory
@time ξ = pronto(θ, x0, η0, τ) # optimal trajectory

##
preview(ξ.x, (1,3))
preview(η0.x, (1,3))
preview(η.x, (1,3))
preview(ξ.x, (2,4))
preview(ξ.u, 1)


opts = Options(
    projection_alg = Rosenbrock23(),
    show_substeps = true,
    show_ξ = true,
    previewfxn = ξ -> preview(ξ.x, (1,3)),
)
# 