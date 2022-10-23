# PRONTO.jl v0.3.0-dev
module PRONTO
include("kernel.jl")







# pronto(θ::Kernel, x0, T/dt, θ, xg, ug)
# pronto(M, x0, T/dt, θ, guess(...)...)
function pronto(θ::Kernel{NX,NU},t,args...) where {NX,NU}
    f(M,x,u,t)
end
# fallback: if type is given, creates an instance
pronto(T::DataType, args...) = pronto(T(), args...)



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