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

function inprod(x)
    i = Int(length(x)/2)
    a = x[1:i]
    b = x[i+1:end]
    P = [a*a'+b*b' -(a*b'+b*a');
        a*b'+b*a' a*a'+b*b']
    return P
end

function x_eig(i)
    α = 3
    v = -α/4
    N = 4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct Split2 <: PRONTO.Model{18,1}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end

@define_f Split2 begin
    α = 3
    v = -α/4
    N = 4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    H1 = v*im*Tridiagonal(ones(2N), zeros(2N+1), -ones(2N))
    H2 = v*Tridiagonal(-ones(2N), zeros(2N+1), -ones(2N))
    return mprod(-im*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

@define_l Split2 begin
    kl/2*u'*I*u 
end

@define_m Split2 begin
    P = I(18) - inprod(x_eig(2))
    return 1/2*x'*P*x
end

@define_Q Split2 begin
    x_re = x[1:9]
    x_im = x[10:18]
    ψ = x_re + im*x_im
    return kq*mprod(I(9) - ψ*ψ')
end

@define_R Split2 kr*I(1)

# must be run after any changes to model definition
resolve_model(Split2)

PRONTO.Pf(θ::Split2,α,μ,tf) = SMatrix{18,18,Float64}(I(18)-inprod(α))
PRONTO.γmax(θ::Split2, ζ, τ) = PRONTO.sphere(1, ζ, τ)
PRONTO.preview(θ::Split2, ξ) = [I(9) I(9)]*(ξ.x.^2)

## ----------------------------------- solve the problem ----------------------------------- ##

θ = Split2(kl=0.01, kr=1, kq=1)
t0,tf = τ = (0,1.9)
x0 = SVector{18}(x_eig(1))
# μ = t->SVector{1}(0.5*sin(4*t))
μ = ξ.u
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-4,maxiters=100,show_steps=false);

## ----------------------------------- output the results ----------------------------------- ##
import Pkg
Pkg.activate()

using MAT

# ts = t0:0.001:tf
ts = range(t0,tf,length=2975)
us = [ξ.u(t)[1] for t in ts]
file = matopen("bs_2hk_3dep_1.9T_50ns.mat", "w")
write(file, "Uopt", us)
close(file)