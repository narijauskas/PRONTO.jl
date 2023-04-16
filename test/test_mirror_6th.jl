using PRONTO
using StaticArrays, LinearAlgebra

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end


# ------------------------------- mirror system on eigenstate 6 ------------------------------- ##

@kwdef struct mirror6 <: Model{36,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    ψ0 = zeros(9,1)
    ψ0[2] = 1
    ψf = zeros(9,1)
    ψf[8] = 1
    xf = vec([-ψf;-ψ0;0*ψf;0*ψ0])
    P = I(36)
    1/2 * collect((x-xf)')*P*(x-xf)
end


# ------------------------------- reflect system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 4
    α = 10
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H00 = kron(I(2),H0)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H11 = kron(I(2),H1)   
    H2 = v*Tridiagonal(-ones(2n), zeros(2n+1), -ones(2n))
    H22 = kron(I(2),H2)
    return mprod(-im*ω*(H00 + sin(u[1])*H11 + (1-cos(u[1]))*H22) )*x
end

stagecost(x,u,t,θ) = 1/2 *θ.kl*collect(u')I*u

regR(x,u,t,θ) = θ.kr*I(1)
function regQ(x,u,t,θ)
    θ.kq*I(36)
end


PRONTO.Pf(α,μ,tf,θ::mirror6) = SMatrix{36,36,Float64}(I(36))

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(mirror6, dynamics, stagecost, termcost, regQ, regR)


## ------------------------------- demo: eigenstate 4->4 in 4 ------------------------------- ##

ψ0 = zeros(9,1)
ψ0[2] = 1
ψf = zeros(9,1)
ψf[8] = 1
x0 = SVector{36}(vec([ψ0;ψf;0*ψ0;0*ψf]))
t0,tf = τ = (0,3.98)


θ = mirror6(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(1.0*sin(16*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 100, limitγ = true)

##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_6hk_mirror_4N_3.98T.mat", "w")
write(file, "Uopt", us)
close(file)
