using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

NX = 4
NU = 1
NΘ = 2

@kwdef struct TwoSpin{T} <: PRONTO.Model{NX,NU,NΘ}
    kr::T = 1.0
    kq::T = 1.0
end

@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

# PRONTO.define_f(TwoSpin, (x,u,t,θ)->begin
#     H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
#     H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
#     (H0 + u[1]*H1)*x
# end)

@stage_cost TwoSpin begin
    Rl = [0.01;;]
    1/2 * u'*Rl*u
end

@terminal_cost TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*x'*Pl*x
end

@regulatorQ TwoSpin θ.kq*I(NX)
@regulatorR TwoSpin θ.kr*I(NU)


x = first(@variables x[1:nx(T)])
u = first(@variables u[1:nu(T)])
t = first(@variables t)
θ = symbolic(T)

sym = invokelatest((x,u,t,θ)->(θ.kq*I(4)), x,u,t,θ)
body = PRONTO.def_kernel(sym)
fxn = PRONTO.def_inplace(:Q, T, body, :x, :u, :t)

@lagrangian TwoSpin
# resolve_model(TwoSpin) # must be run after any change to model definitions

# overwrite default behavior of Pf
# PRONTO.Pf(model::TwoSpin,α,μ,tf) = SMatrix{4,4,Float64}(I(4))
PRONTO.Pf(α,μ,tf,θ::TwoSpin) = SMatrix{4,4,Float64}(I(4))

info(PRONTO.as_bold("TwoSpin")*" model ready")


foo = (x,u,t,θ)->begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end