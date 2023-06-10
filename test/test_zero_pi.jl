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
    E0 = 0.0
    E2 = 3.4
    E9 = 8.1
    H0 = 2π*diagm([E0, E2, E9])
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- 3lvl system to eigenstate 2 ------------------------------- ##

@kwdef struct lvl3 <: Model{6,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    P = I(6) - inprod(x_eig(2))
    100/2 * collect(x')*P*x
end


# ------------------------------- 3lvl system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    E0 = 0.0
    E2 = 3.4
    E9 = 8.1
    H0 = 2π*diagm([E0, E2, E9])

    H1 = [0.0103673+3.73069e-16im 0.00010414+0.000263476im 0.00744436-0.0582357im;
         0.00010414-0.000263476im 0.0103349+1.54752e-15im 0.0300912+0.0165778im;
         0.00744436+0.0582357im 0.0300912-0.0165778im 0.0100988-3.76789e-16im]

    return mprod(-im*(H0+u[1]*H1))*x
end


stagecost(x,u,t,θ) = 1/2*θ.kl*collect(u')I*u

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:3]
    x_im = x[4:6]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(3) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::lvl3) = SMatrix{6,6,Float64}(I(6) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(lvl3, dynamics, stagecost, termcost, regQ, regR)

## ------------------------------- demo: Simulation in 10 ------------------------------- ##

ψ0 = [1;0;0]
ψf = [0;1;0]
x0 = SVector{6}(vec([ψ0;0*ψ0]))
xf = SVector{6}(vec([ψf;0*ψf]))

θ = lvl3(kl=0.001, kr=1, kq=1)

t0,tf = τ = (0,10)

μ = @closure t->SVector{1}(10.0)
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_3lvl_10.mat", "w")
write(file, "Uopt", us)
close(file)