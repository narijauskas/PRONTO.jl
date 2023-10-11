using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define helper functions ----------------------------------- ##

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct XGate3 <: PRONTO.Model{12,1}
    kl::Float64 = 0.01
    kq::Float64 = 0.5
end

@define_f XGate3 begin
    E0 = 0.0
    E1 = 1.0
    E2 = 5.0
    H0 = diagm([E0, E1, E2])
    H00 = kron(I(2),H0)
    a1 = 0.1
    a2 = 0.5
    a3 = 0.3
    Ω1 = a1 * u[1]
    Ω2 = a2 * u[1]
    Ω3 = a3 * u[1]
    H1 = [0 Ω1 Ω3; Ω1 0 Ω2; Ω3 Ω2 0]
    H11 = kron(I(2),H1)
    return 2 * π * mprod(-im * (H00 + H11)) * x
end

@define_l XGate3 begin
    kl/2*u'*I*u + kq/2*x'*mprod(diagm([0,0,1,0,0,1]))*x
end

@define_m XGate3 begin
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(12)*(x-xf)
end

@define_Qr XGate3 I(12)
@define_Rr XGate3 I(1)
PRONTO.Pf(θ::XGate3,α,μ,tf) = SMatrix{12,12,Float64}(I(12))

resolve_model(XGate3)

## ----------------------------------- run optimization ----------------------------------- ##

θ = XGate3()
τ = t0,tf = 0,10
ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory

## ----------------------------------- plot results ----------------------------------- ##
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "control input")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "population")

lines!(ax1, ts, [ξ.u(t)[1] for t in ts], linewidth = 2)

lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[7]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[8]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax2, ts, [ξ.x(t)[3]^2+ξ.x(t)[9]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax2, position = :rc)

lines!(ax3, ts, [ξ.x(t)[4]^2+ξ.x(t)[10]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax3, ts, [ξ.x(t)[5]^2+ξ.x(t)[11]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax3, ts, [ξ.x(t)[6]^2+ξ.x(t)[12]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax3, position = :rc)

display(fig)