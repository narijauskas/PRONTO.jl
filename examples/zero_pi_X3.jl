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

NX = 12
NU = 1

@kwdef struct lvl3 <: PRONTO.Model{NX,NU}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end

@dynamics lvl3 begin
    H00 = kron(I(2),H0[1:3,1:3])
    H11 = kron(I(2),H1[1:3,1:3])
    return mprod(-im * (H00 + u[1]*H11)) * x
end

@stage_cost lvl3 begin
    θ.kl/2*u'*I*u 
end

@terminal_cost lvl3 begin
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(12)*(x-xf)
end

@regulatorQ lvl3 θ.kq*I(NX)
@regulatorR lvl3 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(lvl3)

# overwrite default behavior of Pf
PRONTO.Pf(θ::lvl3,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX))

# runtime plots
PRONTO.runtime_info(θ::lvl3, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))

## ----------------------------------- run optimization ----------------------------------- ##

θ = lvl3()
τ = t0,tf = 0,300

ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{NX}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{NU}(0.4*cos(H0[3,3]*t))
φ = open_loop(θ, x0, μ, τ) # guess trajectory
@time ξ = pronto(θ, x0, φ, τ;verbose=3, tol=1e-4) # optimal trajectory

##

ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
open("3lvl_control_1.0.csv", "w") do io
    writedlm(io, us)
end

##




ts = t0:0.001:tf
is = eachindex(ξ.u)
us = [ξ.u(t)[i] for t∈ts, i∈is]
open("5lvl_control_1.0.csv", "w") do io
    writedlm(io, us)
end