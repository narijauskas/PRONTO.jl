## ----------------------------------- module ----------------------------------- ##
module PRONTO
abstract type Model{NX,NU} end

f(M::Model,x,u) = @error "function f undefined for models of type $(typeof(M))"
fx(M::Model,x,u) = @error "function fx undefined for models of type $(typeof(M))"


nx(::Model{NX,NU}) where {NX,NU} = NX
nu(::Model{NX,NU}) where {NX,NU} = NU

function pronto(M::Model{NX,NU},t,args...) where {NX,NU}
    f(M,x,u)
end
# fallback if type is given:
pronto(T::DataType, args...) = pronto(T(), args...)


end # module
## ----------------------------------- module ----------------------------------- ##

using Main.PRONTO
using Main.PRONTO: nx,nu
using Symbolics
using Symbolics: derivative
using FastClosures
include("../src/autodiff.jl")


# define problem type:
NX = 2; NU = 1
struct FooSystem <: PRONTO.Model{NX,NU} end

# define: f,l,p, regulator
f(x,u) = collect(x)

## ----------------------------------- @configure FooSystem ----------------------------------- ##

macro configure(T)
    return quote
        @variables _x[1:nx($T())] _u[1:nu($T())]

        local f = Main.f
        PRONTO.f(::$T,x,u) = f(x,u)

        local fx = jacobian(_x,f,_x,_u; inplace=false)
        PRONTO.fx(::$T,x,u) = fx(x,u) # NX,NX #NOTE: for testing
        
    end
end

@configure FooSystem
# autodiff/model setup

## ----------------------------------- tests ----------------------------------- ##


# run pronto
# pronto(FooSystem, α, μ, parameters...)
M = FooSystem()
PRONTO.f(M,[0,0],[0])
PRONTO.fx(M,[0,0],[0])


