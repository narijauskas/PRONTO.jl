
struct ModelDefError <: Exception
    M::Model
    fxn::Symbol
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.M)
    print(io, 
        "PRONTO.$(e.fxn) is missing a method for the $T model.\n",
        "Please check the $T model definition and then re-run: ",
        "@derive $T\n"
    )
end

# M not included in args
# @fndef
# @argdef
macro fndef(fn, args)
    fn = esc(fn)
    return quote
        # define a default fallback method
        function ($fn)(M::Model, $(args.args...))
            throw(ModelDefError(M,nameof($fn)))
        end
        # define a symbolic rendition
        function ($fn)(M::T) where {T<:Model}
            # @variables θ[1:nθ(M)]
            # @variables t
            # @variables x[1:nx(M)] 
            # @variables u[1:nu(M)]
            # ξ = vcat(x,u)
            # @variables α[1:nx(M)] 
            # @variables μ[1:nu(M)]
            # φ = vcat(α,μ)
            # @variables z[1:nx(M)] 
            # @variables v[1:nu(M)]
            # ζ = vcat(z,v)
            # @variables Pr[1:nx(M),1:nx(M)]
            # @variables Po[1:nx(M),1:nx(M)]
            # @variables ro[1:nx(M)]
            # @variables λ[1:nx(M)]
            $(_symbolics(:T))
            ($fn)(M, $args...)
        end
    end
end

@fndef f    (θ,t,ξ)
@fndef fx   (θ,t,ξ)
@fndef fu   (θ,t,ξ)
@fndef fxx  (θ,t,ξ)
@fndef fxu  (θ,t,ξ)
@fndef fuu  (θ,t,ξ)

@fndef l    (θ,t,ξ)
@fndef lx   (θ,t,ξ)
@fndef lu   (θ,t,ξ)
@fndef lxx  (θ,t,ξ)
@fndef lxu  (θ,t,ξ)
@fndef luu  (θ,t,ξ)

@fndef p    (θ,t,ξ)
@fndef px   (θ,t,ξ)
@fndef pxx  (θ,t,ξ)

@fndef Rr   (θ,t,φ)
@fndef Qr   (θ,t,φ) 

@fndef Pr_t (θ,t,φ,Pr)
@fndef Kr   (θ,t,φ,Pr)

@fndef ξ_t  (θ,t,ξ,φ,Pr)

@fndef λ_t  (θ,t,ξ,φ,Pr,λ)

@fndef Ko1   (θ,t,ξ,Po) 
@fndef Po1_t (θ,t,ξ,Po)

@fndef Ko2   (θ,t,ξ,λ,Po) 
@fndef Po2_t (θ,t,ξ,λ,Po)

@fndef ro1_t  (θ,t,ξ,Po,ro)
@fndef vo1   (θ,t,ξ,ro)

@fndef ro2_t  (θ,t,ξ,λ,Po,ro)
@fndef vo2   (θ,t,ξ,λ,ro)

@fndef ζ1_t  (θ,t,ξ,ζ,Po,ro)
@fndef _v   (θ,t,ξ,ζ,Po,ro)
@fndef ζ2_t  (θ,t,ξ,ζ,λ,Po,ro)

@fndef y_t  (θ,t,ξ,ζ)
@fndef _Dh  (θ,t,φ,ζ,y)
@fndef _D2g (θ,t,φ,ζ,y)

@fndef h_t  (θ,t,ξ)
@fndef φ̂_t  (θ,t,ξ,φ,ζ,φ̂,γ,Pr)
# cost derivatives...
# @fndef y_t

# the "actual" in-place functions used by PRONTO
f!(M::Model,buf,θ,t,ξ) = throw(ModelDefError(M, :f!))
Pr_t!(M::Model,buf,θ,t,φ,Pr) = throw(ModelDefError(M, :Pr_t!))
ξ_t!(M::Model,buf,θ,t,ξ,φ,P) = throw(ModelDefError(M, :ξ_t!))
λ_t!(M::Model,buf,θ,t,ξ,φ,Pr) = throw(ModelDefError(M, :λ_t!))
Po1_t!(M::Model,buf,θ,t,ξ,P) = throw(ModelDefError(M, :Po1_t!))
Po2_t!(M::Model,buf,θ,t,ξ,λ,P) = throw(ModelDefError(M, :Po2_t!))
ro1_t!(M::Model,buf,θ,t,ξ,Po) = throw(ModelDefError(M, :ro1_t!))
ro2_t!(M::Model,buf,θ,t,ξ,λ,Po) = throw(ModelDefError(M, :ro2_t!))
ζ1_t!(M::Model,buf,θ,t,ξ,ζ,Po,ro) = throw(ModelDefError(M, :ζ1_t!))
ζ2_t!(M::Model,buf,θ,t,ξ,ζ,λ,Po,ro) = throw(ModelDefError(M, :ζ2_t!))
y_t!(M::Model,buf,θ,t,ξ,ζ) = throw(ModelDefError(M, :y_t!))
h_t!(M::Model,buf,θ,t,ξ) = throw(ModelDefError(M, :h_t!))
φ̂_t!(M::Model,buf,θ,t,ξ,φ,ζ,φ̂,γ,Pr) = throw(ModelDefError(M, :φ̂_t!))
Ko!(M::Model,buf,θ,t,ξ,Po) = throw(ModelDefError(M, :Ko_t!))
# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(M,buf,...) = @error
# - symbolic generator symbolic(M,f)