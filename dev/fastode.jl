using DifferentialEquations
using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays
# include("idiomatic_parameterized.jl")
using Main.PRONTO: @buffer


## -----------------------------------  ----------------------------------- ##


M = FooSystem()
x = [0,0]
u = [0]
t = 1
θ = nothing
PRONTO.f(M,x,u,t,θ)

##

# @fn Int -> @buffer(NX,NX)

dynamics!(dx,x,(M,u,θ),t) = PRONTO.f!(dx,M,x,u,t,θ)

sln = let
    sln! = solve(ODEProblem(dynamics!, zeros(2), (0.0,2.0),(M,u,θ)))
    buf = @buffer 2
    FunctionWrapper{MArray{Tuple{NX},Float64,1,NX}, Tuple{Float64}}(t->sln!(buf, t))
end


# T = MArray{Tuple{NX},Float64,1,NX}
# maps from t->x::T
struct Solution{T}
    fn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    ode!::SciMLBase.AbstractODESolution
end

(sln::Solution)(t) = sln.fn(t)

function Solution(prob)
    sln = solve(prob)
    nx = size(sln)[1]
    buf = PRONTO.Buffer{Tuple{nx}}()
    fxn = t->sln(buf, t)
    fn = FunctionWrapper{MArray{Tuple{NX},Float64,1,NX}, Tuple{Float64}}(fxn)
    Solution(fn,buf,sln)
end




Base.show(io::IO, sln::Solution) = show(io,typeof(sln))
function Base.show(io::IO, fn::FunctionWrapper)
    print(io, "$(typeof(fn)), $(fn.ptr), $(fn.objptr)")
end


sln1 = Solution(ODEProblem(dynamics!, zeros(2), (0.0,2.0),(M,u,θ)))
sln2 = Solution(ODEProblem(dynamics!, ones(2), (0.0,2.0),(M,u,θ)))








function testfn(fn)
    for ix in 1:10000000
        fn(sin(ix))
    end
end


fw = let f = (t)->PRONTO.f(M,x,u,t,θ)
    FunctionWrapper{SArray{Tuple{NX},Float64,1,NX}, Tuple{Float64}}(f)
end
