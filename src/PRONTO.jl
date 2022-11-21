module PRONTO

using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
export @closure

function lu end
function lu! end
# using LinearAlgebra # don't import lu
import LinearAlgebra
using LinearAlgebra: diagm, Diagonal
using UnicodePlots
using MacroTools
using SparseArrays
using MatrixEquations

using DifferentialEquations
using Symbolics
using Symbolics: derivative

export @derive
export pronto
export info

export @tick,@tock,@clock

export ODE, ODEBuffer
export dae
export preview





# ----------------------------------- 0. preliminaries & helpers ----------------------------------- #
include("helpers.jl")

# S is a Tuple{dims...}
Buffer{S} = MArray{S, Float64}
Buffer{S}() where {S} = zeros(MArray{S, Float64})
export Buffer


# ----------------------------------- 1. model definition ----------------------------------- #
using MacroTools
using MacroTools: @capture

export @model
# export Model
export nx,nu,nθ


abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ


inv!(A) = LinearAlgebra.inv!(LinearAlgebra.cholesky!(Hermitian(A)))


# ----------------------------------- 2. model derivation ----------------------------------- #
# defines all PRONTO.f(M,θ,t,ξ) and PRONTO.f!(M,θ,t,ξ) along with derivatives
include("model.jl")



# ----------------------------------- 3. intermediate operators ----------------------------------- #
# M contains buffers
@inline Ar(M,θ,t,φ) = (fx!(M, M.Ar, M.θ, t, φ); return M.Ar)
@inline Br(M,θ,t,φ) = (fu!(M, M.Br, M.θ, t, φ); return M.Br)
@inline Qr(M,θ,t,φ) = (Qrr!(M, M.Qr, M.θ, t, φ); return M.Qr)
@inline Rr(M,θ,t,φ) = (Rrr!(M, M.Rr, M.θ, t, φ); return M.Rr)

# function Ar(M::Model{NX,NU,NΘ},θ,t,φ) where {NX,NU,NΘ}
#     buf = Buffer{Tuple{NX,NX}}()
#     fx!(M,buf,θ,t,φ)
#     return buf
# end

# function Br(M::Model{NX,NU,NΘ},θ,t,φ) where {NX,NU,NΘ}
#     buf = Buffer{Tuple{NX,NU}}()
#     fu!(M,buf,θ,t,φ)
#     return buf
#     # fu(M,θ,t,φ)
# end

# function Qr(M::Model{NX,NU,NΘ},θ,t,φ) where {NX,NU,NΘ}
#     buf = Buffer{Tuple{NX,NX}}()
#     Qrr!(M,buf,θ,t,φ)
#     return buf
# end

regulator(B,P,R) = R \ collect(B')*P


riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q


# generate naive, generic Kr, and dPr_dt
function Kr(M, θ, t, φ, Pr)
    M.Kr .= Rr(M,θ,t,φ) \ (Br(M,θ,t,φ)'*Pr)
    return M.Kr
end

function Krr(M::Model{NX,NU,NΘ}, θ, t, φ, Pr) where {NX,NU,NΘ}
    buf = Buffer{Tuple{NU,NX}}()
    buf .= Rr(M,θ,t,φ) \ (Br(M,θ,t,φ)'*Pr)
    return buf
end
# dPr_dt!(M::Model, out, θ, t, φ, Pr) = out .= riccati(Ar(M,θ,t,φ), Kr(M,θ,t,φ,Pr), Pr, Qr(M,θ,t,φ), Rr(M,θ,t,φ))
# include("C:/Users/mantas/AppData/Local/Temp/jl_56RGhMZEBm.jl")
function dPr_dt!(M, out, θ, t, φ, Pr)
    fill!(out, 0)
    A = Ar(M, θ, t, φ)
    B = Br(M, θ, t, φ)
    R = Diagonal(Rr(M, θ, t, φ))
    # K = Kr(M, θ, t, φ, Pr)
    out .-= A'*Pr
    out .-= Pr*A
    out .-= Qr(M, θ, t, φ)
    # out .+= K'*R*K
    out .+= (R\(B'Pr))'R*(R\(B'Pr))
end
# Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_dt!(M,dPr,θ,t,φ(t),Pr)

Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_auto(dPr, Ar(M, θ, t, φ(t)), Br(M, θ, t, φ(t)), Pr, Qr(M, θ, t, φ(t)), Rr(M, θ, t, φ(t)))
# ----------------------------------- 4. ode equations ----------------------------------- #
function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ; tol = 1e-5, maxiters = 20) where {NX,NU,NΘ}
    Pr_f = diagm(ones(NX))
    Pr = ODE(Pr_ode, Pr_f, (tf,t0), (M,θ,φ), Buffer{Tuple{NX,NX}}())
end



# ----------------------------------- 5. ode solutions ----------------------------------- #
include("odes.jl")

# ----------------------------------- 6. trajectories ----------------------------------- #



# ----------------------------------- * buffer type ----------------------------------- #
# ----------------------------------- * timing ----------------------------------- #




# ----------------------------------- PRONTO loop ----------------------------------- #

# ----------------------------------- guess functions ----------------------------------- #
include("guess.jl")

end