
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
        function ($fn)(M::Model)
            @variables θ[1:nθ(M)]
            @variables t
            @variables x[1:nx(M)] 
            @variables u[1:nu(M)]
            ξ = vcat(x,u)
            @variables α[1:nx(M)] 
            @variables μ[1:nu(M)]
            φ = vcat(α,μ)
            @variables z[1:nx(M)] 
            @variables v[1:nu(M)]
            ζ = vcat(z,v)
            @variables Pr[1:nx(M),1:nx(M)]
            @variables Po[1:nx(M),1:nx(M)]
            @variables ro[1:nx(M)]
            @variables λ[1:nx(M)]
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

@fndef Ko   (θ,t,ξ,Po) 
@fndef Po_t (θ,t,ξ,Po)

@fndef λ_t  (θ,t,ξ,φ,Pr,λ)

@fndef ro_t  (θ,t,ξ,Po,ro)
@fndef vo   (θ,t,ξ,ro)

@fndef ζ_t  (θ,t,ξ,Po,ro)
@fndef _v   (θ,t,ξ,Po,ro)

# cost derivatives...
# @fndef y_t

# the "actual" in-place functions used by PRONTO
f!(M::Model,buf,θ,t,ξ) = throw(ModelDefError(M, :f!))
Pr_t!(M::Model,buf,θ,t,φ,Pr) = throw(ModelDefError(M, :Pr_t!))
ξ_t!(M::Model,buf,θ,t,ξ,φ,P) = throw(ModelDefError(M, :ξ_t!))
λ_t!(M::Model,buf,θ,t,ξ,φ,Pr) = throw(ModelDefError(M, :λ_t!))
Po_t!(M::Model,buf,θ,t,ξ,P) = throw(ModelDefError(M, :Po_t!))
ro_t!(M::Model,buf,θ,t,ξ,Po) = throw(ModelDefError(M, :ro_t!))

# FUTURE: for each function and signature, macro-define:
# - default function f(M,...) = @error
# - default inplace f!(M,buf,...) = @error
# - symbolic generator symbolic(M,f)