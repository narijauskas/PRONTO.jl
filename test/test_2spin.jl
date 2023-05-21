using PRONTO
using StaticArrays, LinearAlgebra

function mprod(x) 
    Re = I(2) 
    Im = [0 -1; 1 0] 
    M = kron(Re,real(x)) + kron(Im,imag(x)); 
    return M 
end

function inprod(x) 
    i = Int(length(x)/2) 
    a = x[1:i] 
    b = x[i+1:end] 
    P = [a*a'+b*b' -(a*b'+b*a'); a*b'+b*a' a*a'+b*b'] 
    return P
end

function x_eig(i) 
    H0 = [0 1;1 0] 
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end

@kwdef struct Spin2 <: PRONTO.Model{4,1,3} 
    kl::Float64 # stage cost gain 
    kr::Float64 # regulator r gain 
    kq::Float64 # regulator q gain
end

function termcost(x,u,t,θ) 
    P = I(4) - inprod(x_eig(1)) 
    1/2 * collect(x')*P*x
end

function dynamics(x,u,t,θ) 
    H0 = [0 1;1 0] 
    H1 = [0 -im;im 0] 
    return mprod(-im*(H0 + u[1]*H1) )*x
end

stagecost(x,u,t,θ) = 1/2 *θ[1]*collect(u')I*u

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ) 
    x_re = x[1:2] 
    x_im = x[3:4] 
    ψ = x_re + im*x_im 
    θ.kq*mprod(I(2) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::Spin2) = SMatrix{4,4,Float64}(I(4) - α*α')

PRONTO.generate_model(Spin2, dynamics, stagecost, termcost, regQ, regR)

x0 = SVector{4}(x_eig(2))
t0,tf = τ = (0,10)
θ = Spin2(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)

@time ξ = pronto(θ,x0,φ,τ; tol = 1e-5, maxiters = 50, limitγ = true)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_2spin_10T.mat", "w")
write(file, "Uopt", us)
close(file)