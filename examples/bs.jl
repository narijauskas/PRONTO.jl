using PRONTO
using StaticArrays, LinearAlgebra

N = 4
NX = 2*(2*N+1)
NU = 1

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
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr ≡ θ[2] and θ.kq ≡ θ[3]


## ------------------------------- beam splitter to eigenstate 2 ------------------------------- ##

@kwdef struct Bs2 <: PRONTO.Model{NX,NU}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


@dynamics Bs2 begin
    α = 10 
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    H1 = v*im*Tridiagonal(ones(2N), zeros(2N+1), -ones(2N))
    H2 = v*Tridiagonal(-ones(2N), zeros(2N+1), -ones(2N))
    return mprod(-im*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

@stage_cost Bs2 begin
    θ.kl/2*u'*I*u 
end

@terminal_cost Bs2 begin
    P = I(NX) - inprod(x_eig(2))
    return 1/2*x'*P*x
end

@regulatorQ Bs2 begin
    x_re = x[1:2N+1]
    x_im = x[2N+2:NX]
    ψ = x_re + im*x_im
    return θ.kq*mprod(I(2N+1) - ψ*ψ')
end

@regulatorR Bs2 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(Bs2)

# overwrite default behavior of Pf
PRONTO.Pf(θ::Bs2,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX)-α*α')

# runtime plots
PRONTO.runtime_info(θ::Bs2, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))


## ------------------------------- demo: eigenstate 1->2 in 10s ------------------------------- ##


x0 = SVector{NX}(x_eig(1))
xf = SVector{NX}(x_eig(2))
t0,tf = τ = (0,10)


θ = Bs2(kl=0.01, kr=1, kq=1)
μ = t->SVector{NU}(0.4*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, limitγ = true, verbosity=2)

##

using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_nu2_10_1.0.mat","w")
write(file,"Uopt",us)
close(file)

## ------------------------------- beam splitter to eigenstate 2 ------------------------------- ##

@kwdef struct Bs4 <: PRONTO.Model{NX,NU}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


@dynamics Bs4 begin
    α = 10 
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    H1 = v*im*Tridiagonal(ones(2N), zeros(2N+1), -ones(2N))
    H2 = v*Tridiagonal(-ones(2N), zeros(2N+1), -ones(2N))
    return mprod(-im*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

@stage_cost Bs4 begin
    θ.kl/2*u'*I*u 
end

@terminal_cost Bs4 begin
    P = I(NX) - inprod(x_eig(4))
    return 1/2*x'*P*x
end

@regulatorQ Bs4 begin
    x_re = x[1:2N+1]
    x_im = x[2N+2:NX]
    ψ = x_re + im*x_im
    return θ.kq*mprod(I(2N+1) - ψ*ψ')
end

@regulatorR Bs4 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(Bs4)

# overwrite default behavior of Pf
PRONTO.Pf(θ::Bs4,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX)-α*α')

# runtime plots
PRONTO.runtime_info(θ::Bs4, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))


## ------------------------------- demo: eigenstate 1->4 in 10s ------------------------------- ##

x0 = SVector{NX}(x_eig(1))
xf = SVector{NX}(x_eig(4))
t0,tf = τ = (0,1.5)


θ = Bs4(kl=0.01, kr=1, kq=1)
μ = t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true, verbosity=1)


##
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_nu4_1.5T_1.0.mat","w")
write(file,"Uopt",us)
close(file)