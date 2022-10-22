## ----------------------------------- module ----------------------------------- ##
module PRONTO

using Symbolics
using Symbolics: derivative
include("../src/autodiff.jl")
include("../src/functors.jl")

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


# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro configure(M, NΘ=0)
    T = :(Main.$M)
    return quote
        @variables vx[1:nx($T())] 
        @variables vu[1:nu($T())] 
        @variables vt
        @variables vθ[1:$NΘ]

        local f = Main.f # NX
        # PRONTO.f(::$T,x,u,t,θ) = f(x,u,t,θ)

        local f! = inplace(f,vx,vu,vt,vθ)
        let buf = @buffer nx($T()) 
            # neat behavior, but I don't love how it shares memory...
            # buffer memory should be shared on a per-instance basis
            PRONTO.f(::$T,x,u,t,θ) = (f!(buf,x,u,t,θ); return SArray(buf))
        end

        #NOTE: for testing
        local fx = jacobian(vx,f,vx,vu,vt,vθ; inplace=false)
        PRONTO.fx(::$T,x,u,t,θ) = fx(x,u,t,θ) # NX,NX
    end
end


# struct A
#     α # refs to the correct trajectories
#     μ # refs to the correct trajectories
#     buf # internal buffer?
# end
# (a::A)(θ,t) = a.α
# A holds α and μ as internal type parameters?

# function PRONTO.f(::M,x,u,t,θ)
#     f!(buf,x,u,t,θ)
#     return buf
# end


# fx! = model.fx!; _Ar = Buffer{Tuple{NX,NX}}()
# Ar = @closure (t)->fx!(_Ar,α(t),μ(t))
# # Ar = @closure (t)->(fx!(_Ar,α(t),μ(t)); return _Ar)

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