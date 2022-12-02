# PRONTO.jl dev_0.4
module PRONTO

using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures
export @closure

using LinearAlgebra
using UnicodePlots
using MacroTools
using SparseArrays
using MatrixEquations

# using OrdinaryDiffEq
using DifferentialEquations
using Symbolics
using Symbolics: derivative
using SymbolicUtils.Code

using ThreadTools

export @derive
export pronto
export info

export @tick,@tock,@clock

export ODE, Buffer
export dae
export preview





# ----------------------------------- 0. preliminaries & typedefs ----------------------------------- #


using MacroTools
using MacroTools: @capture

# export @model
export Model
export nx,nu,nθ


abstract type Model{NX,NU,NΘ} <: FieldVector{NΘ,Float64} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ


# not used
inv!(A) = LinearAlgebra.inv!(LinearAlgebra.cholesky!(Hermitian(A)))


# ----------------------------------- 1. helpers ----------------------------------- #
include("helpers.jl")


#YO: can I actually deprecate this? :)
views(::Model{NX,NU,NΘ},ξ) where {NX,NU,NΘ} = (@view ξ[1:NX]),(@view ξ[NX+1:end])


include("codegen.jl") # takes derivatives
include("odes.jl") # ODE solution handling


# defines all PRONTO.f(M,θ,t,ξ) and PRONTO.f!(M,θ,t,ξ) along with derivatives
#TODO: #MAYBE: deprecate
include("model.jl")



# ----------------------------------- 2. model functions ----------------------------------- #
# missing implementations provided by codegen


# or just throw a method error?
struct ModelDefError <: Exception
    θ::Model
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.M)
    print(io, "PRONTO is missing method definitions for the $T model.\n")
end


#TODO: cleanup, don't dispatch on model
Ar(θ,α,μ,t) = throw(ModelDefError(θ))
Br(θ,α,μ,t) = throw(ModelDefError(θ))
Qr(θ,α,μ,t) = throw(ModelDefError(θ))
Rr(θ,α,μ,t) = throw(ModelDefError(θ))
f(θ,x,u,t) = throw(ModelDefError(θ))

# Ar!(out,θ,α,μ,t) = throw(ModelDefError(θ))
# Br!(out,θ,α,μ,t) = throw(ModelDefError(θ))
# Qr!(out,θ,α,μ,t) = throw(ModelDefError(θ))
# Rr!(out,θ,α,μ,t) = throw(ModelDefError(θ))

f!(dx,θ,x,u,t) = throw(ModelDefError(θ))
Kr(θ,α,μ,Pr,t) = Rr(θ,α,μ,t)\(Br(θ,α,μ,t)'Pr)








# ----------------------------------- 3. ode functions ----------------------------------- #
include("guess.jl") #TODO: merge these in




riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q

function dPr_dt!(dPr,Pr,(θ,α,μ),t)#(M, out, θ, t, φ, Pr)
    riccati!(dPr,Ar(θ,α(t),μ(t),t),Kr(θ,α(t),μ(t),Pr,t),Pr,Qr(θ,α(t),μ(t),t),Rr(θ,α(t),μ(t),t))
end

# forced
function dx_dt_ol!(dx,x,(θ,μ),t)
    u = μ(t)
    f!(dx,θ,x,u,t)
end

# regulated
function dx_dt!(dx,x,(θ,α,μ,Pr),t)
    u = μ(t) - Kr(θ,α(t),μ(t),Pr(t),t)*(x-α(t))
    f!(dx,θ,x,u,t)
end
export u_ol,u_cl
# u_ol(θ,μ,t) = μ(t)
# u_cl(θ,x,α,μ,Pr,t) = μ - Kr(θ,α,μ,Pr,t)*(x-α)

# ----------------------------------- pronto loop ----------------------------------- #

export dx_dt!
export dx_dt_ol!


# solves for x(t),u(t)
function pronto(θ::Model{NX,NU,NΘ}, x0::StaticVector, α, μ, (t0,tf);
            tol = 1e-5,
            maxiters = 20) where {NX,NU,NΘ}
   
    Pr_f = diagm(ones(NX)) #TODO: generalize
    # Prf = SizedMatrix{NX,NX}(I(nx(θ)))
    Pr = ODE(dPr_dt!, Prf, (tf,t0), (θ,α,μ), Size(NX,NX))

    # x0 is always the same
    x = ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))

    # ξ = ODE(ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))

end






# ----------------------------------- ?? ----------------------------------- #


# M contains buffers
# @inline Ar(M,θ,t,φ) = (fx!(M, M.Ar, M.θ, t, φ); return M.Ar)
# @inline Br(M,θ,t,φ) = (fu!(M, M.Br, M.θ, t, φ); return M.Br)
# @inline Qr(M,θ,t,φ) = (Qrr!(M, M.Qr, M.θ, t, φ); return M.Qr)
# @inline Rr(M,θ,t,φ) = (Rrr!(M, M.Rr, M.θ, t, φ); return M.Rr)

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
# function Kr(M, θ, t, φ, Pr)
#     mul!(out, Diagonal(R)\B', Pr)
#     M.Kr .= Diagonal(Rr(M,θ,t,φ)) \ (Br(M,θ,t,φ)'*Pr)
#     return M.Kr
# end


# # dPr_dt!(M::Model, out, θ, t, φ, Pr) = out .= riccati(Ar(M,θ,t,φ), Kr(M,θ,t,φ,Pr), Pr, Qr(M,θ,t,φ), Rr(M,θ,t,φ))
# # include("C:/Users/mantas/AppData/Local/Temp/jl_56RGhMZEBm.jl")
# function dPr_dt!(dPr,Pr,(M,θ,φ),t)#(M, out, θ, t, φ, Pr)
#     riccati!(dPr,Ar(M,θ,t,φ),Kr(M,θ,t,φ,Pr),Pr,Qr(M,θ,t,φ),Rr(M,θ,t,φ))
# end
# # Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_dt!(M,dPr,θ,t,φ(t),Pr)

# Pr_ode(dPr,Pr,(M,θ,φ),t) = dPr_auto(dPr, Ar(M, θ, t, φ(t)), Br(M, θ, t, φ(t)), Pr, Qr(M, θ, t, φ(t)), Rr(M, θ, t, φ(t)))
# # ----------------------------------- 4. ode equations ----------------------------------- #
# function pronto(M::Model{NX,NU,NΘ}, θ, t0, tf, x0, u0, φ; tol = 1e-5, maxiters = 20) where {NX,NU,NΘ}
#     Pr_f = diagm(ones(NX))
#     Pr = ODE(dPr_dt!, Pr_f, (tf,t0), (M,θ,φ), Buffer{Tuple{NX,NX}}())
#     ξ = ODE(ξ_ode, [x0;u0], (t0,tf), (M,θ,φ,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))

# end



# ----------------------------------- 5. ode solutions ----------------------------------- #
# include("odes.jl")

# ----------------------------------- 6. trajectories ----------------------------------- #


# ----------------------------------- PRONTO loop ----------------------------------- #

# ----------------------------------- guess functions ----------------------------------- #

end