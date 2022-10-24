# PRONTO.jl v0.3.0-dev
module PRONTO
include("kernels.jl")

using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
using DifferentialEquations


# ----------------------------------- ode solution handling ----------------------------------- #


#MArray{S,T,N,L}
BufferType(S...) = MArray{Tuple{S...}, Float64, length(S), prod(S)}

# maps t->v::T
struct Solution{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    sln::SciMLBase.AbstractODESolution
end

(sln::Solution)(t) = sln.fxn(t)

# T = BufferType(S...)
function Solution(prob, T)
    sln = solve(prob)
    buf = T(undef)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->sln(buf, t))
    Solution(fxn,buf,sln)
end

#FUTURE: show size, length, time span, solver method?
Base.show(io::IO, sln::Solution) = show(io,typeof(sln))

# this might be type piracy... but prevents the obscenely long error messages
function Base.show(io::IO, fn::FunctionWrapper)
    print(io, "$(typeof(fn)), $(fn.ptr), $(fn.objptr)")
end







# ----------------------------------- main ----------------------------------- #





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



end # module