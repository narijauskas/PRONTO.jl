using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

NX = 4
NU = 1

@kwdef struct Spin2 <: PRONTO.Model{NX,NU}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


@dynamics Spin2 begin
    H0 = [1 0;0 -1]
    H1 = [0 -im;im 0]
    H2 = [0 1;1 0]
    return mprod(-im * (H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2)) * x
end


@stage_cost Spin2 begin
    θ.kl/2*(u'*I*u) 
end

@terminal_cost Spin2 begin
    P = [1 0 0 0;
         0 0 0 0;
         0 0 1 0;
         0 0 0 0]
    return 1/2*(x'*P*x)
end

@regulatorQ Spin2 begin
    x_re = x[1:2]
    x_im = x[3:4]
    ψ = x_re + im*x_im
    return θ.kq*mprod(I(2) - ψ*ψ')
end

@regulatorR Spin2 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(Spin2)

# overwrite default behavior of Pf
PRONTO.Pf(θ::Spin2,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX)-α*α')

# runtime plots
PRONTO.runtime_info(θ::Spin2, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))


## ------------------------------- demo: |0⟩ -> |1⟩ in 10 ------------------------------- ##

x0 = SVector{NX}([1 0 0 0])
xf = SVector{NX}([0 1 0 0])
t0,tf = τ = (0,10)


θ = Spin2()
μ = t->SVector{NU}(0.4*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-8, maxiters = 50, limitγ = true, verbosity=1)

## ----------------------------------- output results as MAT ----------------------------------- ##

using MAT

ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
file = matopen("Uopt_10_1.0.mat","w")
write(file,"Uopt",us)
close(file)