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


abstract type Model{NX,NU,NΘ} <: FieldVector{NΘ,Float64} end

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

regulator(B,P,R) = Diagonal(R)\B'P

riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q
# M.buf_nx_nx - could this cause problems, eg. with threading?


function riccati!(out,A,K,P,Q,R)
    # fill!(out, 0) # reset the output (which is reused by the ODE solver)
    # # maybe not the most efficient way to do this, but prevents numerical instabilities
    # copy!(out, Q)
    out .= .- Q
    mul!(out, A', P, -1, 1) # -A'P
    mul!(out, P, A, -1, 1) # -P*A
    # can we more efficiently solve: P'B*(R\B'P) ?
    out .+= K'*R*K
end

# generate naive, generic Kr, and dPr_dt
function Kr(M, θ, t, φ, Pr)
    mul!(out, Diagonal(R)\B', Pr)
    M.Kr .= Diagonal(Rr(M,θ,t,φ)) \ (Br(M,θ,t,φ)'*Pr)
    return M.Kr
end


# dPr_dt!(M::Model, out, θ, t, φ, Pr) = out .= riccati(Ar(M,θ,t,φ), Kr(M,θ,t,φ,Pr), Pr, Qr(M,θ,t,φ), Rr(M,θ,t,φ))
# include("C:/Users/mantas/AppData/Local/Temp/jl_56RGhMZEBm.jl")
function dPr_dt!(dPr,Pr,(M,θ,φ),t)#(M, out, θ, t, φ, Pr)
    riccati!(dPr,Ar(M,θ,t,φ),Kr(M,θ,t,φ,Pr),Pr,Qr(M,θ,t,φ),Rr(M,θ,t,φ))
end
# Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_dt!(M,dPr,θ,t,φ(t),Pr)

Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_auto(dPr, Ar(M, θ, t, φ(t)), Br(M, θ, t, φ(t)), Pr, Qr(M, θ, t, φ(t)), Rr(M, θ, t, φ(t)))
# ----------------------------------- 4. ode equations ----------------------------------- #
function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ; tol = 1e-5, maxiters = 20) where {NX,NU,NΘ}
    Pr_f = diagm(ones(NX))
    Pr = ODE(dPr_dt!, Pr_f, (tf,t0), (M,θ,φ), Buffer{Tuple{NX,NX}}())
    ξ = ODE(ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))

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