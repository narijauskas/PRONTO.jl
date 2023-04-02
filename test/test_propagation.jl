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
    n = 5
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- split system to eigenstate 8 ------------------------------- ##

@kwdef struct propagation <: Model{22,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost8(x,u,t,θ)
    P = 0
    1/2 * collect(x')*P*x
end


# ------------------------------- split system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 5
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    return mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

stagecost(x,u,t,θ) = 1/2 * (θ[1]*collect(u')I*u + 0.3*collect(x')*mprod(diagm([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]))*x)


regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:11]
    x_im = x[12:22]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(11) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::propagation) = SMatrix{22,22,Float64}(I(22) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(propagation, dynamics, stagecost, termcost8, regQ, regR)


## ------------------------------- demo: eigenstate 1->8 in 10 ------------------------------- ##

x0 = SVector{22}([0 0 0 1.0 0 0 0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0])
t0,tf = τ = (0,1.94)


θ = propagation(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(0.2*sin(16.1948*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)


##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_Prop_1.94T.mat", "w")
write(file, "Uprop", us)
close(file)
