# PRONTO.jl v0.3.0-dev
module PRONTO

using Symbolics
using Symbolics: derivative
include("../src/autodiff.jl")
include("../src/functors.jl")

abstract type Model{NX,NU} end
#MAYBE: abstract type Kernel{NX,NU} end

f(M::Model,x,u,t) = @error "function f undefined for models of type $(typeof(M))"
f!(buf,M::Model,x,u,t) = @error "function f! undefined for models of type $(typeof(M))"
fx(M::Model,x,u,t) = @error "function fx undefined for models of type $(typeof(M))"

# "funtion PRONTO.fx is not defined for kernel type $(typeof(θ))"
# "ensure `f(...)` is correctly defined and then run `@configure T`"
# need to know: model type T, function name (eg. fx), function origin (eg. f)

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


# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro configure(M, NΘ=0)
    T = :(Main.$M)
    return quote
        @variables vx[1:nx($T())] 
        @variables vu[1:nu($T())] 
        @variables vt
        @variables vθ[1:$NΘ]

        local f = Main.f # NX
        PRONTO.f(θ::$T,x,u,t) = f(θ,x,u,t)

        local f! = inplace(f,vθ,vx,vu,vt)
        PRONTO.f!(buf,θ::$T,x,u,t) = (f!(buf,θ,x,u,t); return buf)

        #NOTE: for testing
        local fx = jacobian(vx,f,vθ,vx,vu,vt; inplace=false)
        PRONTO.fx(θ::$T,x,u,t) = fx(θ,x,u,t) # NX,NX
    end
end


end # module


# ----------------------------------- dependencies ----------------------------------- #

using Main.PRONTO # import?
using Main.PRONTO: nx,nu
using Main.PRONTO: @configure




## ----------------------------------- problem setup ----------------------------------- ##

# define problem type:
NX = 2; NU = 1
struct FooSystem <: PRONTO.Model{NX,NU} end

# define: f,l,p, regulator
# fn(x,u,t,θ)
f(x,u,t,θ) = collect(x) .+ t

# autodiff/model setup
@configure FooSystem

## ----------------------------------- tests ----------------------------------- ##


# run pronto
# pronto(FooSystem, α, μ, parameters...)
M = FooSystem()
x = [0,0]
u = [0]
t = 1
θ = nothing
PRONTO.f(M,x,u,t,θ)
PRONTO.fx(M,x,u,t,θ)



## ----------------------------------- change model ----------------------------------- ##

# oh, but now I want to add some parameters
f(x,u,t,θ) = θ[1]*collect(x) .+ t*θ[2]

# re-autodiff
@configure FooSystem 2

θ = [2,3]
PRONTO.f(M,x,u,t,θ)