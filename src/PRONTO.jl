# PRONTO.jl v0.3.0-dev
module PRONTO
# include("kernels.jl")
using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using FastClosures

using DifferentialEquations
using Symbolics
using Symbolics: derivative

export @derive
export nx,nu,nθ

export Buffer
export Solution
export Trajectory

# generates symbolic variables for model M
macro symvars(M)
    return esc(quote
        @variables x[1:nx($M)] 
        @variables u[1:nu($M)] 
        @variables t
        @variables θ[1:nθ($M)]
        @variables Pr[1:nx($M),1:nx($M)]
    end)
end

# ----------------------------------- base definitions ----------------------------------- #


abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ


struct ModelDefError <: Exception
    M::Model
    fxn::Symbol
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.M)
    print(io, 
        "PRONTO.$(e.fxn) is missing a method for the $T model.\n",
        "Please check the $T model definition and then re-run: ",
        "@derive $T\n"
    )
end


f(M,x,u,t,θ) = @error "PRONTO.f is missing a method for the $(typeof(M)) model."
fx(M,x,u,t,θ) = @error "PRONTO.fx is missing a method for the $(typeof(M)) model."
fu(M,x,u,t,θ) = @error "PRONTO.fu is missing a method for the $(typeof(M)) model."
fxx(M,x,u,t,θ) = @error "PRONTO.fxx is missing a method for the $(typeof(M)) model."

# l(M,x,u,t,θ)
# p(M,x,u,t,θ)

Rr(M,x,u,t,θ) = @error "PRONTO.Rr is missing a method for the $(typeof(M)) model."
Qr(M,x,u,t,θ) = @error "PRONTO.Qr is missing a method for the $(typeof(M)) model."
Kr(M::Model,x,u,t,θ,P) = throw(ModelDefError(M, :Kr))



f!(buf,M,x,u,t,θ) = @error "PRONTO.f! is missing a method for the $(typeof(M)) model."
Kr!(buf,M,x,u,t,θ,P) = @error "PRONTO.Kr! is missing a method for the $(typeof(M)) model."
dPr!(M,buf,x,u,t,θ,P) = nothing
dPr(M,x,u,t,θ,P) = nothing

#TODO: Kr
#TODO: Pt

ξt!(M::Model,buf,t,θ,x,u,α,μ,P) = throw(ModelDefError(M, :ξt!))


# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(M,buf,...) = @error
# - symbolic generator symbolic(M,f)


riccati(A,K,P,Q,R) = -A'P - P*A + K'R*K - Q


# ----------------------------------- symbolics & autodiff ----------------------------------- #

macro symfunc(fn,args)
    fn = esc(fn)
    return quote
        function ($fn)(M::Model)
            @variables x[1:nx(M)] 
            @variables u[1:nu(M)] 
            @variables t
            @variables θ[1:nθ(M)]
            @variables Pr[1:nx(M),1:nx(M)]
            ($fn)($args...)
        end
    end
end

@symfunc Kr (M,x,u,t,θ,Pr)



function build(f, args...)
    f_sym = cat(Base.invokelatest(f, args...); dims=1)
    f_ex = build_function(f_sym, args...)
    return eval.(f_ex)
end

function jacobian(dx, f, args...)
    f_sym = Base.invokelatest(f, args...)

    # symbolic derivatives
    fx_sym = cat(
        map(1:length(dx)) do i
            map(f_sym) do f
                derivative(f, dx[i])
            end
        end...; dims = ndims(f_sym)+1)

    # return build_function(fx_sym, args...)
    fx_ex = build_function(fx_sym, args...)
    return eval.(fx_ex)
end


struct Jacobian
    dx
end
(J::Jacobian)(f, args...) = jacobian(J.dx, f, args...)


# ----------------------------------- model derivation ----------------------------------- #



# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro derive(T)
    T = esc(T) # make sure we use the local context
    return quote
        @info "starting $($T) model derivation"
        # load user definitions
        local Rr = $(esc(:(Rr)))
        local Qr = $(esc(:(Qr)))
        local f = $(esc(:(f)))
        local l = $(esc(:(l)))
        local p = $(esc(:(p)))

        # define symbolics for derivation
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())] 
        @variables α[1:nx($T())] 
        @variables μ[1:nu($T())] 
        @variables t
        @variables θ[1:nθ($T())]
        @variables Pr[1:nx($T()),1:nx($T())]
        Jx,Ju = Jacobian.([x,u])

        # derive models
        local f,f! = build(f,x,u,t,θ)
        local fx,fx! = Jx(f,x,u,t,θ)
        local fu,fu! = Ju(f,x,u,t,θ)
        local fxx,fxx! = Jx(fx,x,u,t,θ)


        local Kr,Kr! = build(x,u,t,θ,Pr) do x,u,t,θ,Pr
            Rr(x,u,t,θ)\(fu(x,u,t,θ)'*collect(Pr))
        end

        local dPr,dPr! = build(x,u,t,θ,Pr) do x,u,t,θ,Pr
            riccati(fx(x,u,t,θ), Kr(x,u,t,θ,Pr), collect(Pr), Qr(x,u,t,θ), Rr(x,u,t,θ))
        end

        local ξt,ξt! = build(t,θ,x,u,α,μ,Pr) do t,θ,x,u,α,μ,Pr
            vcat(
                f(x,u,t,θ),
                collect(μ) - Kr(x,u,t,θ,Pr)*(collect(x)-collect(α)) - collect(u)
            )
        end




        # add functions to PRONTO - only at this point do we care about dispatch on the first arg
        PRONTO.Rr(M::$T,x,u,t,θ) = Rr(x,u,t,θ)
        PRONTO.Qr(M::$T,x,u,t,θ) = Qr(x,u,t,θ)
        PRONTO.f(M::$T,x,u,t,θ) = f(x,u,t,θ) # NX
        PRONTO.f!(buf,M::$T,x,u,t,θ) = f!(buf,x,u,t,θ)
        # PRONTO.f!(buf,M::$T,x,u,t,θ) = (f!(buf,x,u,t,θ); return buf) # NX
        PRONTO.fx(M::$T,x,u,t,θ) = fx(x,u,t,θ) # NX,NX
        PRONTO.fu(M::$T,x,u,t,θ) = fu(x,u,t,θ) # NX,NU
        PRONTO.fxx(M::$T,x,u,t,θ) = fxx(x,u,t,θ) # NX,NX
        PRONTO.Kr(M::$T,x,u,t,θ,P) = Kr(x,u,t,θ,P) # NU,NX
        PRONTO.Kr!(buf,M::$T,x,u,t,θ,P) = Kr!(buf,x,u,t,θ,P) # NU,NX
        PRONTO.dPr!(M::$T,buf,x,u,t,θ,P) = dPr!(buf,x,u,t,θ,P) # NX,NX
        PRONTO.dPr(M::$T,x,u,t,θ,P) = dPr(x,u,t,θ,P) # NX,NX
        PRONTO.ξt!(M::$T,t,θ,x,u,α,μ,P) = ξt!(t,θ,x,u,α,μ,P) # NX+NU
        

        @info "$($T) model derivation complete!"
    end
end

# where φ=(α,μ)::Trajectory{NX,NU}
function Pr_ode(dPr, Pr,(M,φ,θ), t)
    dPr!(M,dPr,φ(t)...,t,θ,Pr)
end

# ----------------------------------- ode solution handling ----------------------------------- #

#TODO: abstract type AbstractBuffer{T}

# maps t->x::T
struct Buffer{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    fn!::Function
end

(buf::Buffer)(t) = buf.fxn(t)

function Buffer(fn!, N::Vararg{Int})
    T = MArray{Tuple{N...}, Float64, length(N), prod(N)}
    buf = T(undef)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->(fn!(buf, t); return buf))
    Buffer(fxn,buf,fn!)
end



# maps t->x::T
struct Solution{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    sln::SciMLBase.AbstractODESolution
end

(sln::Solution)(t) = sln.fxn(t)

# T = BufferType(S...)
function Solution(prob, N::Vararg{Int})
    sln = solve(prob)
    T = MArray{Tuple{N...}, Float64, length(N), prod(N)}
    buf = T(undef)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->(sln(buf, t); return buf))
    Solution(fxn,buf,sln)
end





# maps t->ξ=(x,u)::(TX,TU)
struct Trajectory{TX,TU}
    x::FunctionWrapper{TX, Tuple{Float64}}
    u::FunctionWrapper{TU, Tuple{Float64}}
    xbuf::TX
    ubuf::TU
    sln::SciMLBase.AbstractODESolution
end

(ξ::Trajectory)(t) = (ξ.x(t), ξ.u(t))

function Trajectory(prob, NX::Int, NU::Int)
    sln = solve(prob) # solves for ξ(t)

    TX = MArray{Tuple{NX...}, Float64, length(NX), prod(NX)}
    xbuf = TX(undef)
    x = FunctionWrapper{TX, Tuple{Float64}}(t->(sln(xbuf,t;idxs=1:NX); return xbuf))

    TU = MArray{Tuple{NU...}, Float64, length(NU), prod(NU)}
    ubuf = TU(undef)
    u = FunctionWrapper{TU, Tuple{Float64}}(t->(sln(ubuf,t;idxs=NX+1:NU); return ubuf))

    Trajectory(x,u,xbuf,ubuf,sln)
end


Base.show(io::IO, buf::Buffer) = show(io,typeof(buf))
Base.show(io::IO, sln::Solution) = show(io,typeof(sln))
#FUTURE: show size, length, time span, solver method?
Base.show(io::IO, trj::Trajectory) = show(io,typeof(trj))
#FUTURE: show size, length, time span, solver method?

# this might be type piracy... but prevents the obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper{T,A}) where {T,A}
    print(io, "FunctionWrapper: $A -> $T $(fn.ptr)")
end







# ----------------------------------- main ----------------------------------- #

# Pr = Solution(ODEProblem(Pr_ode, zeros(2), (0.0,2.0),(M,φ,θ)))

# function Pr_ode(dPr, Pr, (?), t)
#     dPr!(dP,α,μ,t,θ)
# end


# Kr(t,P) = ... 



# function pronto(θ,x0,t0,tf,αg,μg)
 
#     Pr = solution(ODEProblem)

# end
# fx(θ,x,u,t)'*P - P*fx(θ,x,u,t)

# function riccati!(dP, P, (Ar,Br,Rr,Qr), t)
#     # mul!(Kr, Rr(t)\Br(t)', P)
#     # Kr = Rr(t)\Br(t)'*P
#     dP .= -Ar(t)'*P - P*Ar(t) + Kr(t,P)'*Rr(t)*Kr(t,P) - Qr(t)
#     #NOTE: dP is symmetric, as should be P
# end


# # pronto(θ::Kernel, x0, T/dt, θ, xg, ug)
# # pronto(M, x0, T/dt, θ, guess(...)...)
# function pronto(θ::Kernel{NX,NU},t,args...) where {NX,NU}
#     f(M,x,u,t)
# end
# # fallback: if type is given, creates an instance
# pronto(T::DataType, args...) = pronto(T(), args...)

include("utils.jl")


end # module