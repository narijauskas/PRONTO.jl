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
    N = 3
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct SplitHold <: Model{28,1}
    kl::Float64 # stage cost gain
end

@define_f SplitHold begin
    ω = 1.0
    n = 3
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

@define_l SplitHold begin
    kl/2*u'*I*u 
end

@define_m SplitHold begin
    ψ0 = x_eig(1)[1:7]
    ψ1 = x_eig(2)[1:7]
    xf = vec([ψ0;ψ1;0*ψ0;0*ψ1])
    P = I(28)
    return 1/2 * (x-xf)'*P*(x-xf)
end


@define_Q SplitHold I(28)
@define_R SplitHold I(1)

resolve_model(SplitHold)


PRONTO.preview(θ::SplitHold, ξ) = ξ.u
PRONTO.Pf(θ::SplitHold, αf, μf, tf) = SMatrix{28,28,Float64}(I(28))

## ----------------------------------- solve the problem ----------------------------------- ##

ψ3 = x_eig(4)[1:7]
ψ4 = x_eig(5)[1:7]
x0 = SVector{28}(vec([ψ3;ψ4;0*ψ3;0*ψ4]))
t0,tf = τ = (0,2.8)


θ = SplitHold(kl=0.01)
μ = t->SVector{1}(1.0*sin(4*t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-3,maxiters=50);

##
import Pkg
Pkg.activate()

using MAT

ts = t0:0.001:tf
us = [ξ.u(t)[1] for t in ts]
file = matopen("split_hold_2.8T_3N.mat", "w")
write(file, "Uopt", us)
close(file)