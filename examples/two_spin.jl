using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct Qubit <: Model{4,1}
    kl::Float64 = 0.01
end

@define_f Qubit begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@define_l Qubit begin
    1/2*u'*kl*u
end

@define_m Qubit begin
    Pl = [1 0 0 0;0 0 0 0;0 0 1 0;0 0 0 0]
    1/2*x'*Pl*x
end

@define_Qr Qubit I(4)
@define_Rr Qubit I(1)
PRONTO.Pf(θ::Qubit, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))

resolve_model(Qubit)

## ----------------------------------- solve the problem ----------------------------------- ##

θ = Qubit() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [1.0, 0.0, 0.0, 0.0] # initial state
xf = @SVector [0.0, 1.0, 0.0, 0.0] # final state
μ = t->SVector{1}(0.4*sin(t)) # open loop input μ(t)
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory


## ----------------------------------- plot the results ----------------------------------- ##
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "quantum state")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "control input")

lines!(ax1, ts, [ξ.x(t)[1] for t in ts], linewidth = 2, label = "Re(ψ1)")
lines!(ax1, ts, [ξ.x(t)[2] for t in ts], linewidth = 2, label = "Re(ψ2)")
lines!(ax1, ts, [ξ.x(t)[3] for t in ts], linewidth = 2, label = "Im(ψ1)")
lines!(ax1, ts, [ξ.x(t)[4] for t in ts], linewidth = 2, label = "Im(ψ2)")
fig[1, 2] = Legend(fig, ax1)
lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[3]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[4]^2 for t in ts], linewidth = 2, label = "|1⟩")
fig[2, 2] = Legend(fig, ax2)
lines!(ax3, ts, [ξ.u(t)[1] for t in ts], linewidth = 2)


display(fig)