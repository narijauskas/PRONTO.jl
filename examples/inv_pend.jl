using PRONTO
using PRONTO: SymModel
using StaticArrays
using LinearAlgebra

@kwdef struct InvPend <: Model{2,1}
    L::Float64 = 2 # length of pendulum (m)
    g::Float64 = 9.81 # gravity (m/s^2)
end

@define_f InvPend begin
[
    x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
]
end

@define_l InvPend 1/2*x'*I(2)*x + 1/2*u'*I(1)*u


@define_m InvPend begin
    P = [
            1 0;
            0 1;
        ]
    1/2*x'*P*x
end

@define_Q InvPend diagm([10, 1])
@define_R InvPend diagm([1e-3])

resolve_model(InvPend)

PRONTO.runtime_info(θ::InvPend, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.x; color=PRONTO.manto_colors))

##

θ = InvPend() 
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]

α = t->xf
μ = t->u0
η = closed_loop(θ,x0,α,μ,τ)

ξ,data = pronto(θ,x0,η,τ; maxiters=100, tol=1e-4);

##
using GLMakie

fig = Figure()
ax = Axis(fig[1,1],xlabel="iterations",ylabel="-Dg",title="descent",yscale=log10)
y = -data.Dh
lines!(ax,1:length(y),y)
hlines!(ax,1e-4,length(y),color=:red,linestyle=:dash)
display(fig)

save("inv_pend_descent.png",fig)