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


function x_eig(i,θ)
    n = 5
    v = -θ.α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- bs to 6th eigenstate ------------------------------- ##

@kwdef struct Split6 <: Model{22,1,4}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
    α::Float64 # depth of lattice
end


function termcost6(x,u,t,θ)
    P = I(22) - inprod(x_eig(6))
    1/2 * collect(x')*P*x
end


# ------------------------------- bs system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 5
    v = -θ.α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    return mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

stagecost(x,u,t,θ) = 1/2 *θ.kl*collect(u')I*u


regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:11]
    x_im = x[12:22]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(11) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::Split6) = SMatrix{22,22,Float64}(I(22) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(Split6, dynamics, stagecost, termcost6, regQ, regR)


## ------------------------------- demo: eigenstate 1->8 in 10 ------------------------------- ##

x0 = SVector{22}(x_eig(1))
t0,tf = τ = (0,1.5)


θ = Split6(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, limitγ = true)


##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_6hk_20V_1.5T.mat", "w")
write(file, "Uopt", us)
close(file)
