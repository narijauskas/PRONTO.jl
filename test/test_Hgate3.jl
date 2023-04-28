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
    E0 = 0
    E1 = 1
    E2 = 5
    H0 = diagm([E0, E1, E2])
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- hgate3 system to eigenstate 2 ------------------------------- ##

@kwdef struct hgate3 <: Model{12,1,4}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
    T::Float64 # time horizon T
end


function termcostH(x,u,t,θ)
    ψ1 = [1;1;0]/sqrt(2)
    ψ2 = [1;-1;0]/sqrt(2)
    xf = vec([ψ1;ψ2;0*ψ1;0*ψ2])
    P = I(12)
    1/2 * collect((x-xf)')*P*(x-xf)
end


# ------------------------------- hgate3 system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    E0 = 0
    E1 = 1
    E2 = 5
    H0 = diagm([E0, E1, E2])
    H00 = kron(I(2),H0)
    a1 = 0.1
    a2 = 0.5
    a3 = 0.3
    Ω1 = a1*u[1]
    Ω2 = a2*u[1]
    Ω3 = a3*u[1]
    H1 = [0 Ω1 Ω3;Ω1 0 Ω2;Ω3 Ω2 0]
    H11 = kron(I(2),H1)
    return 2*π*mprod(-im*(H00+H11))*x
end


stagecost(x,u,t,θ) = 1/2* (θ.kl* collect(u')I*u + 0.3*collect(x')*mprod(diagm([0,0,1,0,0,1]))*x)

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    θ.kq*I(12)
end

PRONTO.Pf(α,μ,tf,θ::hgate3) = SMatrix{12,12,Float64}(I(12))

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(hgate3, dynamics, stagecost, termcostH, regQ, regR)


## ------------------------------- demo: Xgate in 10 ------------------------------- ##

ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))

θ = hgate3(kl=0.01, kr=1, kq=1, T=10.0)

t0,tf = τ = (0,θ.T)

μ = @closure t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_Hgate_10T.mat", "w")
write(file, "Uopt", us)
close(file)