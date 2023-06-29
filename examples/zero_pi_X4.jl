using DelimitedFiles
using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

H0 = readdlm("H0.csv", ',', ComplexF64)
H1 = readdlm("H1.csv", ',', ComplexF64)


function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

##

NX = 16
NU = 1

@kwdef struct lvl4 <: PRONTO.Model{NX,NU}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end

@dynamics lvl4 begin
    H00 = kron(I(2),H0[1:4,1:4])
    H11 = kron(I(2),H1[1:4,1:4])
    return mprod(-im * (H00 + u[1]*H11)) * x
end

@stage_cost lvl4 begin
    θ.kl/2*u'*I*u + 0.1*x'*mprod(diagm([0,0,0,1,0,0,0,1]))*x
end

@terminal_cost lvl4 begin
    ψ1 = [1;0;0;0]
    ψ2 = [0;1;0;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(NX)*(x-xf)
end

@regulatorQ lvl4 θ.kq*I(NX)
@regulatorR lvl4 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(lvl4)

# overwrite default behavior of Pf
PRONTO.Pf(θ::lvl4,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX))

# runtime plots
PRONTO.runtime_info(θ::lvl4, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))

##
θ = lvl4()
τ = t0,tf = 0,300

ψ1 = [1;0;0;0]
ψ2 = [0;1;0;0]
x0 = SVector{NX}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{NU}(0.4*cos(H0[3,3]*t))
φ = open_loop(θ, x0, μ, τ) # guess trajectory
@time ξ = pronto(θ, x0, φ, τ;verbose=1, tol=1e-4, maxiters=6) # optimal trajectory

##
ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
open("4lvl_control_1.0.csv", "w") do io
    writedlm(io, us)
end