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
    E11 = 9.1
    E18 = 12.54
    H0 = diagm([E0, E2, E9, E11, E18])
    w = eigvecs(collect(H0)) 
    x_eig = kron([1;0],w[:,i])
end


# ------------------------------- 5lvl system to eigenstate 2 ------------------------------- ##

@kwdef struct lvl5 <: Model{10,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    P = I(10) - inprod(x_eig(2))
    1/2 * collect(x')*P*x
end


# ------------------------------- 5lvl system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    E0 = 0.0
    E2 = 3.4
    E9 = 8.1
    E11 = 9.1
    E18 = 12.54
    H0 = diagm([E0, E2, E9, E11, E18])

    H1 = [0.0103673+3.73069e-16im 0.00010414+0.000263476im 0.00744436-0.0582357im 0.717964-0.908256im -0.0189716-0.0447698im;
         0.00010414-0.000263476im 0.0103349+1.54752e-15im 0.0300912+0.0165778im -0.000539014-0.000929565im 1.09378-0.0267062im;
         0.00744436+0.0582357im 0.0300912-0.0165778im 0.0100988-3.76789e-16im -0.0121773-0.0073288im -0.00422462+0.00246372im;
         0.717964+0.908256im -0.000539014+0.000929565im -0.0121773+0.0073288im 0.0149801-5.5961e-16im 0.00124781-0.00227833im;
         -0.0189716+0.0447698im 1.09378+0.0267062im -0.00422462-0.00246372im 0.00124781+0.00227833im 0.0248604+3.19226e-16im]

    return mprod(-im*(H0+u[1]*H1))*x
end


stagecost(x,u,t,θ) = 1/2*θ.kl*collect(u')I*u + 0.0*collect(x')*mprod(diagm([0, 0, 0, 1, 0]))*x + 0.0*collect(x')*mprod(diagm([0, 0, 0, 0, 1]))*x

regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:5]
    x_im = x[6:10]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(5) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::lvl5) = SMatrix{10,10,Float64}(I(10) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(lvl5, dynamics, stagecost, termcost, regQ, regR)

## ------------------------------- demo: Simulation in 10 ------------------------------- ##

ψ0 = [1;0;0;0;0]
x0 = SVector{10}(vec([ψ0;0*ψ0]))

θ = lvl5(kl=0.01, kr=1, kq=1)

t0,tf = τ = (0,10)

μ = @closure t->SVector{1}(2.0*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, limitγ = true)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_5lvl_10.mat", "w")
write(file, "Uopt", us)
close(file)