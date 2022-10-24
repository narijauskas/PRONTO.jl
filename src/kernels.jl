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

fxx(M::Model,x,u,t,θ) = @error "function PRONTO.fxx is not defined for model type $(typeof(M))"

# l(M,x,u,t,θ)
# p(M,x,u,t,θ)

Rr(M::Model,x,u,t,θ) = @error "PRONTO.Rr is not defined for model type $(typeof(M))"
Qr(M::Model,x,u,t,θ) = @error "PRONTO.Qr is not defined for model type $(typeof(M))"
# Pr(M::Model,x,u,t,θ) = @error "PRONTO.Pr is not defined for model type $(typeof(M))"
Kr(M::Model,x,u,t,θ,P) = @error "PRONTO.Kr is not defined for model type $(typeof(M))"
Kr!(buf,M::Model,x,u,t,θ,P) = @error "PRONTO.Kr! is not defined for model type $(typeof(M))"

#TODO: Kr
#TODO: Pt

# ModelDefinitionError
# $(typeof(M)) is missing a method definition for "PRONTO.fx"
# "ensure `f(...)` is correctly defined and then run `@configure T`"
# need to know: model type T, function name (eg. fx), function origin (eg. f)






# ----------------------------------- symbolics & autodiff ----------------------------------- #

function build(f, args...)
    f_sym = cat(Base.invokelatest(f, args...); dims=1)
    f_ex = build_function(f_sym, args...)
    return eval.(f_ex)
end

function jacobian(dx, f, args...)
    f_sym = Base.invokelatest(f, args...)

    # symbolic derivatives
    fx_sym = cat(
        map(1:length(dx)) do i
            map(f_sym) do f
                derivative(f, dx[i])
            end
        end...; dims = ndims(f_sym)+1)

    # return build_function(fx_sym, args...)
    fx_ex = build_function(fx_sym, args...)
    return eval.(fx_ex)
end


struct Jacobian
    dx
end
(J::Jacobian)(f, args...) = jacobian(J.dx, f, args...)
  

# ----------------------------------- model derivation ----------------------------------- #


# loads definitions for model M into pronto from autodiff based on current definitions in Main
macro derive(T)
    T = esc(T) # make sure we use the local context
    return quote

        # load user definitions
        local Rr = $(esc(:(Rr)))
        local Qr = $(esc(:(Qr)))
        local f = $(esc(:(f)))
        local l = $(esc(:(l)))
        local p = $(esc(:(p)))

        # define symbolics for derivation
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())] 
        @variables t
        @variables θ[1:nθ($T())]
        @variables P[1:nx($T()),1:nx($T())]
        Jx,Ju = Jacobian.([x,u])

        # derive models
        local f,f! = build(f,x,u,t,θ)
        local fx,fx! = Jx(f,x,u,t,θ)
        local fu,fu! = Ju(f,x,u,t,θ)
        local fxx,fxx! = Jx(fx,x,u,t,θ)

        # local fx = allocating(jacobian(x,f,x,u,t,θ; inplace=false))
        # local fx = build(Jx(f,x,u,t,θ))
        
        # local _Kr = (x,u,t,θ,P) -> (Rr(x,u,t,θ)\(fu(x,u,t,θ)'*collect(P)))
        # local Kr,Kr! = build(_Kr,x,u,t,θ,P)
        local Kr,Kr! = build(x,u,t,θ,P) do x,u,t,θ,P
            Rr(x,u,t,θ)\(fu(x,u,t,θ)'*collect(P))
        end
        # local Kr = inplace(x,u,t,θ,P; inplace=false) do (x,u,t,θ,P)
        #     Rr(x,u,t,θ)\(fu(x,u,t,θ)'P)
        # end


        # add functions to PRONTO - only at this point do we care about dispatch on the first arg
        @inline PRONTO.Rr(M::$T,x,u,t,θ) = Rr(x,u,t,θ)
        @inline PRONTO.Qr(M::$T,x,u,t,θ) = Qr(x,u,t,θ)
        @inline PRONTO.f(M::$T,x,u,t,θ) = f(x,u,t,θ) # NX
        @inline PRONTO.f!(buf,M::$T,x,u,t,θ) = f!(buf,x,u,t,θ)
        # @inline PRONTO.f!(buf,M::$T,x,u,t,θ) = (f!(buf,x,u,t,θ); return buf) # NX
        @inline PRONTO.fx(M::$T,x,u,t,θ) = fx(x,u,t,θ) # NX,NX
        @inline PRONTO.fu(M::$T,x,u,t,θ) = fu(x,u,t,θ) # NX,NU
        @inline PRONTO.fxx(M::$T,x,u,t,θ) = fxx(x,u,t,θ) # NX,NX
        @inline PRONTO.Kr(M::$T,x,u,t,θ,P) = Kr(x,u,t,θ,P) # NU,NX
        @inline PRONTO.Kr!(buf,M::$T,x,u,t,θ,P) = Kr!(buf,x,u,t,θ,P) # NU,NX
    end
end