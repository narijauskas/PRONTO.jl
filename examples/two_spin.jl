using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef

NX = 4
NU = 1
NΘ = 2

@kwdef struct TwoSpin{T} <: PRONTO.Model{NX,NU,NΘ}
    kr::T = 1
    kq::T = 1
end

## ----------------------------------- generate solver kernel ----------------------------------- ##

@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@stage_cost TwoSpin begin
    Rl = [0.01;;]
    1/2 * collect(u')*Rl*u
end

@terminal_cost TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*collect(x')*Pl*x
end

@regulatorQ TwoSpin θ.kq*I(NX)
@regulatorR TwoSpin θ.kr*I(NU)
@lagrangian TwoSpin


# PRONTO.symtype(TwoSpin)

# struct Two2Spin{T} <: PRONTO.Model{NX,NU,NΘ}
#     kr::T
#     kq::T
# end

# struct Two3Spin{T} <: PRONTO.Model{NX,NU,NΘ}
#     kr::T
#     kq::SVector{T,4}
# end


# #YO: this will need to be updated to support non-scalar parameters
# symindex(T, name) = findfirst(isequal(name), fieldnames(T))
# symfields(T,θ) = Tuple(θ[symindex(T, name)] for name in fieldnames(T))

# function SymbolicModel(T)
#     @variables θ[1:nθ(T)]
#     (; zip(fieldnames(TwoSpin), symfields(TwoSpin, collect(θ)))...)
# end

## ----------------------------------- model definition ----------------------------------- ##
# function dynamics(x,u,t,θ)
#     H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
#     H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
#     (H0 + u[1]*H1)*x
# end

# Rreg(x,u,t,θ) = θ[1]*I(NU)
# Qreg(x,u,t,θ) = θ[2]*I(NX)

# function stagecost(x,u,t,θ)
#     Rl = [0.01;;]
#     1/2 * collect(u')*Rl*u
# end

# function termcost(x,u,t,θ)
#     Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
#     1/2*collect(x')*Pl*x
# end

# # @model 4 1 TwoSpin begin
#     kr::Float64 = 1
#     kq::Float64 = 1
# end

# foo = quote
#     struct Name{T} <: Model{T,4,1,2}
#         kr::T
#         kq::T
#         vec::SVector{4,T}
#     end
# end

# foo = quote
#     kr::T
#     kq::T
#     vec::SVector{4,T}
# end



# struct 𝚯TwoSpin <

# θ = TwoSpin{Float64}()
# θ = TwoSpin()




# PRONTO.generate_model(TwoSpin, dynamics, stagecost, termcost, Qreg, Rreg)

# PRONTO.build_f(TwoSpin, dynamics)
# PRONTO.build_l(TwoSpin, stagecost)
# PRONTO.build_p(TwoSpin, termcost)
# PRONTO.build_L(TwoSpin)






# PRONTO.build_f
# PRONTO.build_l
# PRONTO.build_L
# PRONTO.build_p
# PRONTO.build_QR

# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::TwoSpin{T}) where T = SMatrix{4,4,Float64}(I(4))

## ----------------------------------- tests ----------------------------------- ##

θ = TwoSpin{Float64}() # make an instance of the mode.
τ = t0,tf = 0,10

x0 = @SVector [0.0, 1.0, 0.0, 0.0]
xf = @SVector [1.0, 0.0, 0.0, 0.0]
u0 = 0.1
μ = @closure t->SizedVector{1}(u0)
φ = open_loop(θ, xf, μ, τ) # guess trajectory
ξ = pronto(θ, x0, φ, τ) # optimal trajectory
