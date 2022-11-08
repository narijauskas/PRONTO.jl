
struct ModelDefError <: Exception
    M::Model
    fn::Symbol
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.M)
    print(io, 
        "PRONTO.$(e.fn) is missing a method for the $T model.\n",
        "Please check the $T model definition and then re-run: ",
        "@derive $T\n"
    )
end

# turn :(fn) into :(fn!)
_!(ex) = Symbol(String(ex)*"!")

macro _!(ex)
    # :($(Symbol(String(ex)*"!")))
    # Symbol(String($ex)*"!")
    :(Symbol(String($ex)*"!"))
end

# @define f (θ,t,ξ)
macro define(fn, args)
    fn! = esc(_!(fn)) # generates :(fn!)
    fn = esc(fn)
    return quote

        # define a fallback method for fn(M, args...) for undefined M
        function ($fn)(M::Model, $(args.args...))
            throw(ModelDefError(M,nameof($fn)))
        end

        # define a fallback method for fn!(M, buf, args...) for undefined M
        function ($fn!)(M::Model, buf, $(args.args...))
            throw(ModelDefError(M,nameof($fn!)))
        end

        # define PRONTO.fn(M) to show symbolic form
        function ($fn)(M::T) where {T<:Model}
            $(_symbolics(:T))
            ($fn)(M, $args...)
        end

        # does defining PRONTO.fn!(M, buf) make any sense?
    end
end

# @define f(θ,t,ξ)

@define f    (θ,t,ξ)
@define fx   (θ,t,ξ)
@define fu   (θ,t,ξ)
@define fxx  (θ,t,ξ)
@define fxu  (θ,t,ξ)
@define fuu  (θ,t,ξ)

@define l    (θ,t,ξ)
@define lx   (θ,t,ξ)
@define lu   (θ,t,ξ)
@define lxx  (θ,t,ξ)
@define lxu  (θ,t,ξ)
@define luu  (θ,t,ξ)

@define p    (θ,t,ξ)
@define px   (θ,t,ξ)
@define pxx  (θ,t,ξ)

@define Rr   (θ,t,φ)
@define Qr   (θ,t,φ) 

@define Pr_t (θ,t,φ,Pr)
@define Kr   (θ,t,φ,Pr)

@define ξ_t  (θ,t,ξ,φ,Pr)

@define λ_t  (θ,t,ξ,φ,Pr,λ)

@define Ko1   (θ,t,ξ,Po) 
@define Po1_t (θ,t,ξ,Po)

@define Ko2   (θ,t,ξ,λ,Po) 
@define Po2_t (θ,t,ξ,λ,Po)

@define ro1_t  (θ,t,ξ,Po,ro)
@define vo1   (θ,t,ξ,ro)

@define ro2_t  (θ,t,ξ,λ,Po,ro)
@define vo2   (θ,t,ξ,λ,ro)

@define ζ1_t  (θ,t,ξ,ζ,Po,ro)
@define _v   (θ,t,ξ,ζ,Po,ro)
@define ζ2_t  (θ,t,ξ,ζ,λ,Po,ro)

@define y_t  (θ,t,ξ,ζ)
@define _Dh  (θ,t,φ,ζ,y)
@define _D2g (θ,t,φ,ζ,y)

@define h_t  (θ,t,ξ)
@define φ̂_t  (θ,t,ξ,φ,ζ,φ̂,γ,Pr)
# cost derivatives...
# @fndef y_t

# the "actual" in-place functions used by PRONTO
# f!(M::Model,buf,θ,t,ξ) = throw(ModelDefError(M, :f!))
# Pr_t!(M::Model,buf,θ,t,φ,Pr) = throw(ModelDefError(M, :Pr_t!))
# ξ_t!(M::Model,buf,θ,t,ξ,φ,P) = throw(ModelDefError(M, :ξ_t!))
# λ_t!(M::Model,buf,θ,t,ξ,φ,Pr) = throw(ModelDefError(M, :λ_t!))
# Po1_t!(M::Model,buf,θ,t,ξ,P) = throw(ModelDefError(M, :Po1_t!))
# Po2_t!(M::Model,buf,θ,t,ξ,λ,P) = throw(ModelDefError(M, :Po2_t!))
# ro1_t!(M::Model,buf,θ,t,ξ,Po) = throw(ModelDefError(M, :ro1_t!))
# ro2_t!(M::Model,buf,θ,t,ξ,λ,Po) = throw(ModelDefError(M, :ro2_t!))
# ζ1_t!(M::Model,buf,θ,t,ξ,ζ,Po,ro) = throw(ModelDefError(M, :ζ1_t!))
# ζ2_t!(M::Model,buf,θ,t,ξ,ζ,λ,Po,ro) = throw(ModelDefError(M, :ζ2_t!))
# y_t!(M::Model,buf,θ,t,ξ,ζ) = throw(ModelDefError(M, :y_t!))
# h_t!(M::Model,buf,θ,t,ξ) = throw(ModelDefError(M, :h_t!))
# φ̂_t!(M::Model,buf,θ,t,ξ,φ,ζ,φ̂,γ,Pr) = throw(ModelDefError(M, :φ̂_t!))
# Ko!(M::Model,buf,θ,t,ξ,Po) = throw(ModelDefError(M, :Ko_t!))
# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(M,buf,...) = @error
# - symbolic generator symbolic(M,f)