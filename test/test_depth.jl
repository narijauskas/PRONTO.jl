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

@kwdef struct Depth <: Model{22,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost2(x,u,t,θ)
    P = I(22) - inprod(x_eig(2))
    1/2 * (100*collect(x')*P*x)
end


# ------------------------------- split system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    ω = 1.0
    n = 5
    # H = SymTridiagonal(promote([4.0i^2 for i in -n:n], (-collect(u/4))*ones(2n))...)
    H = [100 -u/4 0 0 0 0 0 0 0 0 0;-u/4 64 -u/4 0 0 0 0 0 0 0 0;0 -u/4 36 -u/4 0 0 0 0 0 0 0;0 0 -u/4 16 -u/4 0 0 0 0 0 0;0 0 0 -u/4 4 -u/4 0 0 0 0 0;0 0 0 0 -u/4 0 -u/4 0 0 0 0;0 0 0 0 0 -u/4 4 -u/4 0 0 0;0 0 0 0 0 0 -u/4 16 -u/4 0 0;0 0 0 0 0 0 0 -u/4 36 -u/4 0;0 0 0 0 0 0 0 0 -u/4 64 -u/4;0 0 0 0 0 0 0 0 0 -u/4 100]
    return mprod(-im*ω*H)*x
end

stagecost(x,u,t,θ) = 1/2 *θ[1]*collect((u.-10)')*I*(u.-10)


regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    x_re = x[1:11]
    x_im = x[12:22]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(11) - ψ*ψ')
end

PRONTO.Pf(α,μ,tf,θ::Depth) = SMatrix{22,22,Float64}(I(22) - α*α')

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(Depth, dynamics, stagecost, termcost2, regQ, regR)


## ------------------------------- demo: eigenstate 1->8 in 10 ------------------------------- ##

x0 = SVector{22}(x_eig(1))
t0,tf = τ = (0,5)


θ = Depth(kl=0.001, kr=1, kq=1)
μ = @closure t->SVector{1}(5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 100, limitγ = true)


##
using MAT
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_2hk_5T.mat", "w")
write(file, "Uopt", us)
close(file)
