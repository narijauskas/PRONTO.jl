using Symbolics
using Symbolics: derivative
export @derive
export nx,nu,nθ

abstract type Model{NX,NU,NΘ} end

nx(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NX
nu(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NU
nθ(::Model{NX,NU,NΘ}) where {NX,NU,NΘ} = NΘ

f(M::Model,x,u,t,θ) = @error "function PRONTO.f is not defined for model type $(typeof(M))"
f!(buf,M::Model,x,u,t,θ) = @error "function PRONTO.f! is not defined for model type $(typeof(M))"
fx(M::Model,x,u,t,θ) = @error "function PRONTO.fx is not defined for model type $(typeof(M))"
fu(M::Model,x,u,t,θ) = @error "function PRONTO.fu is not defined for model type $(typeof(M))"

# l(M,x,u,t,θ)
# p(M,x,u,t,θ)

Rr(M::Model,x,u,t,θ) = @error "PRONTO.Rr is not defined for model type $(typeof(M))"
Qr(M::Model,x,u,t,θ) = @error "PRONTO.Qr is not defined for model type $(typeof(M))"
# Pr(M::Model,x,u,t,θ) = @error "PRONTO.Pr is not defined for model type $(typeof(M))"
Kr(M::Model,x,u,t,θ,P) = @error "PRONTO.Kr is not defined for model type $(typeof(M))"

#TODO: Kr
#TODO: Pt

# ModelDefinitionError
# $(typeof(M)) is missing a method definition for "PRONTO.fx"
# "ensure `f(...)` is correctly defined and then run `@configure T`"
# need to know: model type T, function name (eg. fx), function origin (eg. f)






# ----------------------------------- autodiff ----------------------------------- #

# # what do I need?
# fxn -> inplace fxn
# fxn -> simplified fxn
# fxn -> jacobian
# jacobian -> inplace fxn
# jacobian -> hessian
# hessian -> inplace fxn

# problem
# symbolic -> standalone inplace fxn syntax would be clunky due to repeated args




#MAYBE: refactor with functors
#TODO: better document mappings
# Jx = Jacobian(x)
# Hxx = Hessian(x,x) # == Jx∘Jx

# Jx(f, args...; inplace)
# Hxx(f, args...; inplace)

# build_inplace(f, args...)
# build_normal(f, args...)

# function allocating(ex)
#     @eval $(ex[1])
# end

# function inplace(ex)
#     @eval $(ex[2])
# end

function inplace(f, args...; inplace=true)
    f_sym = cat(Base.invokelatest(f, args...); dims=1)
    f_ex = build_function(f_sym, args...)[inplace ? 2 : 1]
    return @eval $f_ex
end



# fx = jacobian(x, f, x, u)
function jacobian(dx, f, args...; inplace = false)
    f_sym = Base.invokelatest(f, args...)

    # symbolic derivatives
    fx_sym = cat(
        map(1:length(dx)) do i
            map(f_sym) do f
                derivative(f, dx[i])
            end
        end...; dims = ndims(f_sym)+1)

    # return build_function(fx_sym, args...)
    fx_ex = build_function(fx_sym, args...)[inplace ? 2 : 1]
    return @eval $fx_ex
end

# fxx = hessian(x, u, f, x, u)
hessian(dx1, dx2, f, args...; inplace = false) = jacobian(dx2, jacobian(dx1, f, args...), args...; inplace)



# ----------------------------------- model derivation ----------------------------------- #


# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro derive(T)
    T = esc(T) # make sure we use the local context
    return quote
        # find local definitions
        local Rr = $(esc(:(Rr)))
        local Qr = $(esc(:(Qr)))
        local f = $(esc(:(f)))

        # define symbolics for derivation
        @variables vx[1:nx($T())] 
        @variables vu[1:nu($T())] 
        @variables vt
        @variables vθ[1:nθ($T())]
        @variables vP[1:nx($T()),1:nx($T())]


        # derive models
        local f! = inplace(f,vx,vu,vt,vθ)
        local fx = jacobian(vx,f,vx,vu,vt,vθ; inplace=false)
        local fu = jacobian(vu,f,vx,vu,vt,vθ; inplace=false)


        # local fx = allocating(jacobian(x,f,x,u,t,θ; inplace=false))
        # local fx = build(Jx(f,x,u,t,θ))
        
        local _Kr = (x,u,t,θ,P) -> (Rr(x,u,t,θ)\(fu(x,u,t,θ)'*collect(P)))
        local Kr = inplace(_Kr,vx,vu,vt,vθ,vP; inplace=false)
        # local Kr = inplace(x,u,t,θ,P; inplace=false) do (x,u,t,θ,P)
        #     Rr(x,u,t,θ)\(fu(x,u,t,θ)'P)
        # end


        # add functions to PRONTO - only at this point do we care about dispatch on the first arg
        PRONTO.Rr(M::$T,x,u,t,θ) = Rr(x,u,t,θ)
        PRONTO.Qr(M::$T,x,u,t,θ) = Qr(x,u,t,θ)

        PRONTO.f(M::$T,x,u,t,θ) = f(x,u,t,θ) # NX
        PRONTO.f!(buf,M::$T,x,u,t,θ) = (f!(buf,x,u,t,θ); return buf) # NX
        PRONTO.fx(M::$T,x,u,t,θ) = fx(x,u,t,θ) # NX,NX
        PRONTO.fu(M::$T,x,u,t,θ) = fu(x,u,t,θ) # NX,NU
        PRONTO.Kr(M::$T,x,u,t,θ,P) = Kr(x,u,t,θ,P) # NU,NX
    end
end