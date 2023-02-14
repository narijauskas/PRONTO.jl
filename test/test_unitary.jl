using PRONTO
using StaticArrays, LinearAlgebra

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end


# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr == θ[2] and θ.kq == θ[3]


# ------------------------------- mirror system on eigenstate 4 ------------------------------- ##

@kwdef struct unit4 <: Model{28,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    ψ0 = zeros(7,1)
    ψ0[2] = 1
    ψf = zeros(7,1)
    ψf[6] = 1
    xf = vec([-ψf;-ψ0;0*ψf;0*ψ0])
    P = I(28)
    1/2 * collect((x-xf)')*P*(x-xf)
end


# ------------------------------- reflect system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 3
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
    θ.kq*I(28)
end


PRONTO.Pf(α,μ,tf,θ::unit4) = SMatrix{28,28,Float64}(I(28))

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(unit4, dynamics, stagecost, termcost, regQ, regR)


## ------------------------------- demo: eigenstate 4->4 in 4 ------------------------------- ##

ψ0 = zeros(7,1)
ψ0[2] = 1
ψf = zeros(7,1)
ψf[6] = 1
x0 = SVector{28}(vec([ψ0;ψf;0*ψ0;0*ψf]))
t0,tf = τ = (0,2.88)


θ = unit4(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(1.0*sin(12*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-3, maxiters = 100, limitγ = true)


##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_4hk_mirror.mat", "w")
write(file, "Uopt", us)
close(file)
