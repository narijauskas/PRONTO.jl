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


function x_eig(i)
    n = 6
    α = 36
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- split system to eigenstate 8 ------------------------------- ##

@kwdef struct Split8 <: Model{26,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost8(x,u,t,θ)
    P = I(26) - inprod(x_eig(8))
    1/2 * collect(x')*P*x
end


# ------------------------------- split system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 6
    α = 36
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    return mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

stagecost(x,u,t,θ) = 1/2 *θ[1]*collect(u')I*u


regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:13]
    x_im = x[14:26]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(13) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::Split8) = SMatrix{26,26,Float64}(I(26) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(Split8, dynamics, stagecost, termcost8, regQ, regR)


## ------------------------------- demo: eigenstate 1->8 in 10 ------------------------------- ##

x0 = SVector{26}(x_eig(1))
t0,tf = τ = (0,2)


θ = Split8(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)


##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_8hk_36V_2T.mat", "w")
write(file, "Uopt", us)
close(file)
