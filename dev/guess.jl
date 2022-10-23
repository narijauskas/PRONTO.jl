
using DifferentialEquations
using FunctionWrappers
using FunctionWrappers: FunctionWrapper
using StaticArrays


NX = 2; NU = 1
struct TestKernel <: PRONTO.Model{NX,NU}
end
f(θ,x,u,t) = [-1;0;;0;-1]*collect(x)

@configure TestKernel 0

θ = TestKernel()
x = [2,1]
u = [0]
t = 0
t0 = 0.0
tf = 5.0
T = (t0,tf)
PRONTO.f(θ,x,u,t)



#MArray{S,T,N,L}
BufferType(S...) = MArray{Tuple{S...}, Float64, length(S), prod(S)}


# maps t->v::T
#MAYBE: Buffer(fxn, S...)
struct Buffer{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
end

(buf::Buffer)(t) = buf.fxn(t)




function Buffer(fxn!::Function, S...)
    T = BufferType(S...)
    Buffer(FunctionWrapper{T, Tuple{Float64}}(fxn!), T(undef))
end



Base.show(io::IO, buf::Buffer) = show(io,typeof(buf))
#FUTURE: show size, length, time span, solver method?
function Base.show(io::IO, fn::FunctionWrapper)
    print(io, "$(typeof(fn)), $(fn.ptr), $(fn.objptr)")
end



# maps t->v::T
struct Solution{T}
    fxn::FunctionWrapper{T, Tuple{Float64}}
    buf::T
    sln::SciMLBase.AbstractODESolution
end

(sln::Solution)(t) = sln.fxn(t)

function Solution(S, prob)
    sln = solve(prob)
    T = BufferType(S...)
    buf = T(undef)
    fxn = FunctionWrapper{T, Tuple{Float64}}(t->sln(buf, t))
    Solution(fxn,buf,sln)
end

Base.show(io::IO, sln::Solution) = show(io,typeof(sln))
#FUTURE: show size, length, time span, solver method?
function Base.show(io::IO, fn::FunctionWrapper)
    print(io, "$(typeof(fn)), $(fn.ptr), $(fn.objptr)")
end


function zi!(dx,x,(θ,u),t)
    PRONTO.f!(dx,θ,x,u,t)
end

prob = ODEProblem(zi!, [2;1], (t0,tf), (θ,u))

αg = Solution((NX,), prob)
μg = let
    T = Buffer(NU)
    buf = T(undef); buf .= 0
    fn = FunctionWrapper{T, Tuple{Float64}}(t->buf)
end