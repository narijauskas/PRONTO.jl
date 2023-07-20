using PRONTO
using FastClosures
using StaticArrays
using LinearAlgebra
using MatrixEquations



@kwdef struct InvPend <: Model{2,1}
    L::Float64 = 2 # length of pendulum (m)
    g::Float64 = 9.81 # gravity (m/s^2)
end

@dynamics InvPend [
    x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
]

@define_l InvPend 1/2*x'*I(2)*x + 1/2*u'*I(1)*u


@define_m InvPend begin
    P = [
            88.0233 39.3414;
            39.3414 17.8531;
        ]
    1/2*x'*P*x
end

@define_Q InvPend diagm([10, 1])
@define_R InvPend diagm([1e-3])

resolve_model(InvPend)

# PRONTO.runtime_info(θ::InvPend, ξ; verbosity) = nothing
# PRONTO.runtime_info(θ::InvPend, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u; color=PRONTO.manto_colors[1]))
PRONTO.runtime_info(θ::InvPend, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x; color=PRONTO.manto_colors))
# Rreg(x,u,t,θ) = diagm([1e-3])
# Qreg(x,u,t,θ) = diagm([10, 1])

# function stagecost(x,u,t,θ)
#     Ql = I(NX)
#     Rl = I(NU)
#     1/2 * collect(x')*Ql*x + 1/2 * collect(u')*Rl*u
# end

# function termcost(x,u,t,θ)
#     Pl = [
#             88.0233 39.3414;
#             39.3414 17.8531;
#         ]
#     1/2*collect(x')*Pl*x
# end

# PRONTO.generate_model(InvPend, dynamics, stagecost, termcost, Qreg, Rreg)
# ##
θ = InvPend()
ξ,data = pronto(θ, x0, ξ0, τ);


##
# θ = InvPend(g=3.71) # on mars
θ = InvPend() # on mars
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]
# smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0

α = t->xf
μ = t->u0
η3 = closed_loop(θ,x0,α,μ,τ)
# closed_loop(θ, x0, α, μ, τ)

η1 = smooth(θ, x0, xf, τ)
η2 = smooth(θ, x0, xf, t->u0*sin(t), τ)
ξ,data = pronto(θ,x0,η3,τ; maxiters=1000);
ξ,data = pronto(θ,x0,η1,τ; maxiters=1000);

##
using GLMakie
n = Observable(1)
fig = Figure()
ax = Axis(fig[1,1])
T = LinRange(τ...,10000)
x1 = @lift [data.ξ[$n].x(t)[1] for t in T]
x2 = @lift [data.ξ[$n].x(t)[2] for t in T]
lines!(ax, T, x1)
lines!(ax, T, x2)
ylims!(ax,-7,7)

u1 = @lift [data.ξ[$n].u(t)[1] for t in T]
ax = Axis(fig[2,1])
lines!(ax, T, u1)
ylims!(ax,-7,7)
display(fig)
##

record(x->n[]=x, fig, "test_mars.mp4", 1:length(data.ξ))

planets = [
    "Sun"=>274.1,
    "Mercury"=>3.703,
    "Venus"=>8.872,
    "Earth"=>9.8067,
    "Moon"=>1.625,
    "Mars"=>3.728,
    "Jupiter"=>25.93,
    "Saturn"=>11.19,
    "Uranus"=>9.01,
    "Neptune"=>11.28,
    "Pluto"=>0.610,
]

# θ = InvPend(g=274.1)
θ = InvPend(g=61.0)
η = smooth(θ, x0, xf, τ)
ξ,data = pronto(θ,x0,η,τ; maxiters=1000, armijo_maxiters=100)

for (planet,g) in planets[6]
    @show θ = InvPend(g=g)
    η = smooth(θ, x0, xf, τ)
    ξ,data = pronto(θ,x0,η,τ; maxiters=1000, armijo_maxiters=100)
    # println("$planet has a gravity of $g m/s^2")
end


Kr = regulator(θ,φ,τ)
φ = projection(θ,x0,φ,Kr,τ)

x0 = @SVector [2π/3;0]
# φ = open_loop(θ,xf,μ,τ)
φ = zero_input(θ,xf,τ)
@time pronto(θ,x0,φ,τ; maxiters=1000, verbose=false)
# M = InvPend()
# θ = Float64[]
# x0 = [2π/3;0]
# u0 = [0.0]
# ξ0 = [x0;u0]
# ξf = [0;0;0]
# t0 = 0.0; tf = 10.0


##
φg = @closure t->ξf
φ = guess_φ(M,θ,ξf,t0,tf,φg)
##
@time ξ = pronto(M,θ,t0,tf,x0,u0,φ; tol = 1e-8, maxiters=100)


##