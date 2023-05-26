using PRONTO
using StaticArrays, LinearAlgebra

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
    n = 4
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr == θ[2] and θ.kq == θ[3]


# ------------------------------- mirror system on eigenstate 4 ------------------------------- ##

@kwdef struct hold4 <: Model{36,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    P = I(18) - inprod(x_eig(1))
    return 1/2 * collect(x')*kron(I(2),P)*x
end


# ------------------------------- reflect system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    n = 4
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H00 = mprod(-im*H0)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H11 = mprod(-im*H1) 
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    H22 = mprod(-im*H2)
    return (kron(I(2),H00) + sin(u[1])*kron(I(2),H11) + (1-cos(u[1]))*kron(I(2),H22)) *x
end

stagecost(x,u,t,θ) = 1/2 *θ.kl*collect(u')I*u

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    # x_re1 = x[1:9]
    # x_re2 = x[10:18]
    # x_im1 = x[19:27]
    # x_im2 = x[28:36]
    # ψ1 = x_re1 + im*x_im1
    # ψ2 = x_re2 + im*x_im2
    # Q1 = I(9) - ψ1*ψ1'
    # Q2 = I(9) - ψ2*ψ2'
    # θ.kq*mprod([Q1 zeros(9,9);zeros(9,9) Q2])
    θ.kq*I(36)
end


PRONTO.Pf(α,μ,tf,θ::hold4) = SMatrix{36,36,Float64}(I(36) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(hold4, dynamics, stagecost, termcost, regQ, regR)


## ------------------------------- demo: eigenstate 4->4 in 4 ------------------------------- ##

ψ0 = ((x_eig(4) - x_eig(5))/sqrt(2))
ψf = ((x_eig(4) + x_eig(5))/sqrt(2))
x0 = SVector{36}(vec([ψ0;ψf]))
t0,tf = τ = (0,5.0)


θ = hold4(kl=0.001, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 100, limitγ = true)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_421_2T.mat", "w")
write(file, "Uopt", us)
close(file)
