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

# get the ith eigenstate
function x_eig(i)
    α = 10 
    v = -α/4
    N = 5
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end


## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct Split4 <: PRONTO.Model{22,1}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end

@define_f Split4 begin
    α = 10 
    v = -α/4
    N = 5
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    H1 = v*im*Tridiagonal(ones(2N), zeros(2N+1), -ones(2N))
    H2 = v*Tridiagonal(-ones(2N), zeros(2N+1), -ones(2N))
    return mprod(-im*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

@define_l Split4 begin
    kl/2*u'*I*u 
end

@define_m Split4 begin
    P = I(22) - inprod(x_eig(4))
    return 1/2*x'*P*x
end

@define_Q Split4 begin
    x_re = x[1:11]
    x_im = x[12:22]
    ψ = x_re + im*x_im
    return kq*mprod(I(11) - ψ*ψ')
end

@define_R Split4 kr*I(1)

# must be run after any changes to model definition
resolve_model(Split4)

PRONTO.Pf(θ::Split4,α,μ,tf) = SMatrix{22,22,Float64}(I(22)-inprod(α))
PRONTO.γmax(θ::Split4, ζ, τ) = PRONTO.sphere(1, ζ, τ)
PRONTO.preview(θ::Split4, ξ) = [I(11) I(11)]*(ξ.x.^2)

## ----------------------------------- solve the problem ----------------------------------- ##
# eigenstate 1->4

θ = Split4(kl=0.01, kr=1, kq=1)
t0,tf = τ = (0,1.5)
x0 = SVector{22}(x_eig(1))
μ = t->SVector{1}(0.4*sin(t))
η = open_loop(θ,x0,μ,τ)
ξ,data = pronto(θ,x0,η,τ;tol=1e-3,maxiters=50,show_steps=true);
