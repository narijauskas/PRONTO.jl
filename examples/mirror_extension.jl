using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef

## ----------------------------------- define helper functions ----------------------------------- ##

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct Mirror4 <: Model{37,1}
    kl::Float64 # stage cost gain
end

@define_f Mirror4 begin
    ω = 1.0
    n = 4
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H00 = kron(I(2),H0)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H11 = kron(I(2),H1)   
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    H22 = kron(I(2),H2)
    return [mprod(-im*ω*(H00 + sin(x[37])*H11 + (1-cos(x[37]))*H22) )*x[1:36];u[1]]
end

@define_l Mirror4 begin
    kl/2*u'*I*u 
end

@define_m Mirror4 begin
    ψ0 = zeros(9,1)
    ψ0[3] = 1
    ψf = zeros(9,1)
    ψf[7] = 1
    xf = vec([-ψf;-ψ0;0*ψf;0*ψ0])
    P = I(36)
    return 1/2 * collect((x[1:36]-xf)')*P*(x[1:36]-xf)
end


@define_Q Mirror4 I(37)
@define_R Mirror4 I(1)

resolve_model(Mirror4)


PRONTO.preview(θ::Mirror4, ξ) = ξ.u
PRONTO.Pf(θ::Mirror4, αf, μf, tf) = SMatrix{37,37,Float64}(I(37))

## ----------------------------------- solve the problem ----------------------------------- ##

ψ0 = zeros(9,1)
ψ0[3] = 1
ψf = zeros(9,1)
ψf[7] = 1
x0 = SVector{37}(vec([ψ0;ψf;0*ψ0;0*ψf;0]))
t0,tf = τ = (0,2.88)


θ = Mirror4(kl=0.001)
μ = t->SVector{1}(1.0*sin(12*t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-3,maxiters=50,show_steps=false);

##
import Pkg
Pkg.activate()

using MAT

ts = t0:0.001:tf
us = [ξ.x(t)[end] for t in ts]
file = matopen("mirror_2.88T_4N_EX_12w.mat", "w")
write(file, "Uopt", us)
close(file)