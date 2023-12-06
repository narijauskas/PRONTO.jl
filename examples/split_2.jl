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

function x_eig(i)
    α = 10 
    v = -α/4
    N = 4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    # x_eig = kron([1;0],w[:,i])
    x_eig = w[:,i]
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct Beam2 <: Model{36,1}
    kl::Float64 # stage cost gain
end

@define_f Beam2 begin
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
    return mprod(-im*ω*(H00 + sin(u[1])*H11 + (1-cos(u[1]))*H22) )*x
end

@define_l Beam2 begin
    kl/2*u'*I*u 
end

@define_m Beam2 begin
    ν3 = x_eig(4)
    ν4 = x_eig(5)
    xf = vec([ν3;ν4;0*ν3;0*ν4])
    P = I(36)
    return 1/2 * collect((x-xf)')*P*(x-xf)
end


@define_Q Beam2 I(36)
@define_R Beam2 I(1)

resolve_model(Beam2)


PRONTO.preview(θ::Beam2, ξ) = ξ.u
PRONTO.Pf(θ::Beam2, αf, μf, tf) = SMatrix{36,36,Float64}(I(36))

## ----------------------------------- solve the problem ----------------------------------- ##

ν3 = x_eig(4)
ν4 = x_eig(5)
ψ0 = (ν3-ν4)/sqrt(2)
ψ1 = (ν3+ν4)/sqrt(2)
x0 = SVector{36}(vec([ψ0;ψ1;0*ψ0;0*ψ1]))
t0,tf = τ = (0,2.88)


θ = Beam2(kl=0.01)
μ = t->SVector{1}(1.0*sin(t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-4,maxiters=50,show_steps=false);

## ----------------------------------- output the results ----------------------------------- ##
import Pkg
Pkg.activate()

using MAT

ts = t0:0.001:tf
ts = range(t0,tf,Int(floor(1566*2.88)))
us = [ξ.u(t)[1] for t in ts]
file = matopen("bs_4_2.88T_4N_1064.mat", "w")
write(file, "Uopt", us)
close(file)