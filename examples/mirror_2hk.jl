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

@kwdef struct Mirror2 <: Model{36,1}
    kl::Float64 # stage cost gain
end

@define_f Mirror2 begin
    ω = 1.0
    n = 4
    α = 3.0
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H00 = kron(I(2),H0)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H11 = kron(I(2),H1)   
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    H22 = kron(I(2),H2)
    return mprod(-im*ω*(H00 + sin(u[1])*H11 + (1-cos(u[1]))*H22) )*x
end

@define_l Mirror2 begin
    kl/2*u'*I*u 
end

@define_m Mirror2 begin
    ψ0 = zeros(9,1)
    ψ0[4] = 1
    ψf = zeros(9,1)
    ψf[6] = -1
    xf = vec([ψf;ψ0;0*ψf;0*ψ0])
    P = I(36)
    return 1/2 * collect((x-xf)')*P*(x-xf)
end


@define_Q Mirror2 I(36)
@define_R Mirror2 I(1)

resolve_model(Mirror2)


PRONTO.preview(θ::Mirror2, ξ) = ξ.u
PRONTO.Pf(θ::Mirror2, αf, μf, tf) = SMatrix{36,36,Float64}(I(36))

## ----------------------------------- solve the problem ----------------------------------- ##

ψ0 = zeros(9,1)
ψ0[4] = 1
ψf = zeros(9,1)
ψf[6] = -1
x0 = SVector{36}(vec([ψ0;ψf;0*ψ0;0*ψf]))
t0,tf = τ = (0,3.93)


θ = Mirror2(kl=0.01)
μ = t->SVector{1}(1.5*sin(4*t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-3,maxiters=50,show_steps=false);

## ----------------------------------- output the results ----------------------------------- ##
import Pkg
Pkg.activate()

using MAT

ts = t0:0.001:tf
us = [ξ.u(t)[1] for t in ts]
file = matopen("mirror_2hk_5.5T.mat", "w")
write(file, "Uopt", us)
close(file)