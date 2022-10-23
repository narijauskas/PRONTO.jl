using BenchmarkTools
using FastClosures

using FunctionWrappers
import FunctionWrappers: FunctionWrapper

# For a function that sends (x1::T1, x2::T2, ...) -> ::TN, you use
# a FunctionWrapper{TN, Tuple{T1, T2, ...}}.
struct TypeStableStruct 
    fun::FunctionWrapper{Float64, Tuple{Float64, Float64}}
    second_arg::Float64
end

(str::TypeStableStruct)(arg) = str.fun(arg, str.second_arg)
wrapper = TypeStableStruct(hypot, 1.0)
wrapper(1.5) == hypot(1.5,1.0)

anon = (x)->hypot(x,1.0)
closure = @closure (x)->hypot(x,1.0)
dispatch(x) = hypot(x,1.0)

@btime hypot(1.5,1.0)
@btime wrapper(1.5)
@btime anon(1.5)
@btime closure(1.5)
@btime dispatch(1.5)
@btime functor(1.5)


struct Functor{F}
    fun::F
    second_arg::Float64
end
(fn::Functor)(arg) = fn.fun(arg, fn.second_arg)

functor = Functor(hypot, 1.0)


cfxn_ptr = @cfunction anon Float64 (Float64,) 
cfxn() = ccall(cfxn_ptr, Float64, (Float64,), Float64)