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

# must be run after any changes to model definition
resolve_model(TwoSpin)
##

# ξ->ξ.x[(1,3)]

# PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x, (1,3)))
function PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1)
    if verbosity >= 1
        println(preview(ξ.x, (1,3); color=PRONTO.manto_colors))
        println(preview(ξ.x, (2,4); color=PRONTO.manto_colors[3:4]))
        println(preview(ξ.u; color=PRONTO.manto_colors))
    end
end

PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))
PRONTO.runtime_info(θ::TwoSpin, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x; color=PRONTO.manto_colors))


# overwrite default behavior of Pf for TwoSpin models
PRONTO.Pf(θ::TwoSpin, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))
# PRONTO.γmax(θ::TwoSpin, ζ, τ) = min(1, 1/maximum(maximum(ζ.x(t) for t in LinRange(τ..., 10000))))

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



## --------------------- plots --------------------- ##
using GLMakie

dt = 0.01
T = t0:dt:tf

fig = Figure()

sl = Slider(fig[3,1:2], range=1:18, startvalue=3)
ix = sl.value

ax = Axis(fig[1,1])
ylims!(ax, (-1,1))

for i in 1:4
    x = @lift [data.ξ[$ix].x(t)[i] for t in T]
    lines!(ax, T, x)
end

ax = Axis(fig[1,2])
u = @lift [data.ξ[$ix].u(t)[1] for t in T]
lines!(ax, T, u)


ax = Axis(fig[2,1])
# ylims!(ax, (-1,1))

for i in 1:4
    z = @lift [data.ξ[$ix].x(t)[i] + data.ζ[$ix].x(t)[i] for t in T]
    lines!(ax, T, z)
end

ax = Axis(fig[2,2])
u = @lift [data.ξ[$ix].u(t)[1] for t in T]
lines!(ax, T, u)
v = @lift [data.ξ[$ix].u(t)[1]+data.ζ[$ix].u(t)[1] for t in T]
lines!(ax, T, v)


display(fig)



##
record(fig, "animated.mp4", 1:18) do jx
    sl.value[] = jx
end