using Symbolics
using Symbolics: derivative

function inplace(f, args...; inplace=true)
    f_sym = cat(Base.invokelatest(f, args...); dims=1)
    f_ex = build_function(f_sym, args...)[inplace ? 2 : 1]
    return @eval $f_ex
end

# fx = jacobian(x, f, x, u)
function jacobian(dx, f, args...; inplace = false)
    f_sym = Base.invokelatest(f, args...)
    # f_sym = f(args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(f_sym) do f
            derivative(f, dx[i])
        end
    end...; dims = ndims(f_sym)+1)
    fx_ex = build_function(fx_sym, args...)[inplace ? 2 : 1]
    return @eval $fx_ex
end

# fxx = hessian(x, u, f, x, u)
hessian(dx1, dx2, f, args...; inplace = false) = jacobian(dx2, jacobian(dx1, f, args...), args...; inplace)






# include("../src/autodiff.jl")
# include("../src/functors.jl")

abstract type Kernel{NX,NU} end

nx(::Kernel{NX,NU}) where {NX,NU} = NX
nu(::Kernel{NX,NU}) where {NX,NU} = NU
# nθ(::Kernel) doesn't actually matter!

f(θ::Kernel,x,u,t) = @error "function PRONTO.f is not defined for kernel type $(typeof(θ))"
f!(buf,θ::Kernel,x,u,t) = @error "function PRONTO.f! is not defined for kernel type $(typeof(θ))"
fx(θ::Kernel,x,u,t) = @error "function PRONTO.fx is not defined for kernel type $(typeof(θ))"

# "funtion PRONTO.fx is not defined for kernel type $(typeof(θ))"
# "ensure `f(...)` is correctly defined and then run `@configure T`"
# need to know: model type T, function name (eg. fx), function origin (eg. f)






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