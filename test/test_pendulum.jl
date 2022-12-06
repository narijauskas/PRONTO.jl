
using PRONTO
using FastClosures
using StaticArrays
using LinearAlgebra
using MatrixEquations


NX = 2
NU = 1
NΘ = 0

struct InvPend <: PRONTO.Model{NX,NU,0}
end

function dynamics(x,u,t,θ)
    g = 9.81
    L = 2
    [x[2],
    g/L*sin(x[1])-u[1]*cos(x[1])/L]
end


Rreg(x,u,t,θ) = diagm([1e-3])
Qreg(x,u,t,θ) = diagm([10, 1])

function stagecost(x,u,t,θ)
    Ql = I(NX)
    Rl = I(NU)
    1/2 * collect(x')*Ql*x + 1/2 * collect(u')*Rl*u
end

function termcost(x,u,t,θ)
    Pl = [
            88.0233 39.3414;
            39.3414 17.8531;
        ]
    1/2*collect(x')*Pl*x
end

PRONTO.generate_model(InvPend, dynamics, stagecost, termcost, Qreg, Rreg)
##

##
let
    g = 9.81
    L = 2

    f = (θ,t,x,u) -> [
        x[2],
        g/L*sin(x[1])-u[1]*cos(x[1])/L
    ]
    
    Ql = I
    Rl = I
    l = (θ,t,x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
    
    Rr = (θ,t,x,u) -> diagm([1e-3])
    Qr = (θ,t,x,u) -> diagm([10, 1])
    
    p = (θ,t,x,u) -> begin
        # P,_ = arec(fx(θ,t,ξ_eq), fu(θ,t,ξ_eq)*inv(Rr(θ,t,x_eq,u_eq))*fu(θ,t,ξ_eq)', Qr(θ,t,x_eq,u_eq))
        # P,_ = arec(fx(θ,t,ξ_eq), fu(θ,t,ξ_eq), luu(θ,t,ξ_eq), lxx(θ,t,ξ_eq), lxu(θ,t,ξ_eq))
        P = [
            88.0233 39.3414;
            39.3414 17.8531;
        ]
        1/2*collect(x)'*P*collect(x)
    end

    @derive InvPend
end



##
θ = InvPend()
τ = t0,tf = 0,10
x0 = @SVector [π;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]
smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
μ = @closure t->u0*sin(t)
α = @closure t->smooth(t, x0, xf, tf)
φ = PRONTO.Trajectory(θ,α,μ);
Kr = regulator(θ,φ,τ)
φ = projection(θ,x0,φ,Kr,τ)
φ = open_loop(θ,xf,μ,τ)
φ = zero_input(θ,x0,τ)
pronto(θ,x0,φ,τ; maxiters=1000, verbose=false)
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
