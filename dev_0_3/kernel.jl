
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

# MAYBE: merge this into build
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

# temporarily required
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

@define dPr_dt (θ,t,φ,Pr)
@define Kr   (θ,t,φ,Pr)

@define dξ_dt  (θ,t,ξ,φ,Pr)

@define dλ_dt  (θ,t,ξ,φ,Pr,λ)

@define Ko_1   (θ,t,ξ,Po) 
@define dPo_dt_1 (θ,t,ξ,Po)

@define Ko_2   (θ,t,ξ,λ,Po) 
@define dPo_dt_2 (θ,t,ξ,λ,Po)

@define dro_dt_1  (θ,t,ξ,Po,ro)
@define vo_1   (θ,t,ξ,ro)

@define dro_dt_2  (θ,t,ξ,λ,Po,ro)
@define vo_2   (θ,t,ξ,λ,ro)

@define dζ_dt_1  (θ,t,ξ,ζ,Po,ro)
@define _v   (θ,t,ξ,ζ,Po,ro)
@define dζ_dt_2  (θ,t,ξ,ζ,λ,Po,ro)

@define dy_dt  (θ,t,ξ,ζ)
@define _Dh  (θ,t,φ,ζ,y)
@define _D2g (θ,t,φ,ζ,y)

@define dh_dt  (θ,t,ξ)
@define dφ̂_dt  (θ,t,ξ,φ,ζ,φ̂,γ,Pr)
