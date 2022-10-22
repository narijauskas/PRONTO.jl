## ----------------------------------- module ----------------------------------- ##
module PRONTO

using Symbolics
using Symbolics: derivative
include("../src/autodiff.jl")

abstract type Model{NX,NU} end

f(M::Model,x,u,t) = @error "function f undefined for models of type $(typeof(M))"
f!(buf,M::Model,x,u,t) = @error "function f! undefined for models of type $(typeof(M))"
fx(M::Model,x,u,t) = @error "function fx undefined for models of type $(typeof(M))"


nx(::Model{NX,NU}) where {NX,NU} = NX
nu(::Model{NX,NU}) where {NX,NU} = NU
# nθ(::Model) doesn't actually matter!


# pronto(M::Model, x0, T/dt, θ, xg, ug)
# pronto(M, x0, T/dt, θ, guess(...)...)
function pronto(M::Model{NX,NU},t,args...) where {NX,NU}
    f(M,x,u,t)
end
# fallback: if type is given, creates an instance
pronto(T::DataType, args...) = pronto(T(), args...)


# loads definitions into pronto from autodiff based on current definitions in Main
macro configure(T)
    return quote
        @variables vx[1:nx(Main.$T())] vu[1:nu(Main.$T())] vt

        local f = Main.f # NX
        PRONTO.f(::Main.$T,x,u,t) = f(x,u,t)

        local f! = inplace(f,vx,vu,vt)
        PRONTO.f!(buf,::Main.$T,x,u,t) = f!(buf,x,u,t)

        #NOTE: for testing
        local fx = jacobian(vx,f,vx,vu,vt; inplace=false)
        PRONTO.fx(::Main.$T,x,u,t) = fx(x,u,t) # NX,NX
    end
end


end # module


# ----------------------------------- dependencies ----------------------------------- #

using Main.PRONTO
using Main.PRONTO: nx,nu
using Main.PRONTO: @configure




## ----------------------------------- problem setup ----------------------------------- ##

# define problem type:
NX = 2; NU = 1
struct FooSystem <: PRONTO.Model{NX,NU} end

# define: f,l,p, regulator
# fn(x,u,t,θ)
f(x,u,t) = collect(x).+t

# autodiff/model setup
@configure FooSystem

## ----------------------------------- tests ----------------------------------- ##


# run pronto
# pronto(FooSystem, α, μ, parameters...)
M = FooSystem()
x = [0,0]
u = [0]
t = 1
buf = similar(x)
PRONTO.f(M,x,u,t)
PRONTO.f!(buf,M,x,u,t)
PRONTO.fx(M,x,u,t)


