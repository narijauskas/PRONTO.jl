using MacroTools
using MacroTools: postwalk, striplines
using Base: invokelatest
using Symbolics
using SparseArrays
using Dates


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

_!(ex) = Symbol(String(ex)*"!")

macro define(fn, args)
    fn! = esc(_!(fn)) # generates :(fn!)
    fn = esc(fn)
    return quote

        # define a fallback method for fn(M, args...) for undefined M
        function ($fn)(M::Model, $(args.args...))
            throw(ModelDefError(M,nameof($fn)))
        end

        # define a fallback method for fn!(M, buf, args...) for undefined M
        function ($fn!)(M::Model, out, $(args.args...))
            throw(ModelDefError(M,nameof($fn!)))
        end

        # define PRONTO.fn(M) to show symbolic form
        # function ($fn)(M::T) where {T<:Model}
        #     $(_symbolics(:T))
        #     ($fn)(M, $args...)
        # end

        # does defining PRONTO.fn!(M, buf) make any sense?
    end
end

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
@define Qr   (θ,t,ξ)
@define Rr   (θ,t,ξ)

#TODO: macro
# f(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(f)))
# f!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(f!)))
# fx(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fx)))
# fx!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fx!)))
# fu(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fu)))
# fu!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fu!)))
# fxx(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fxx)))
# fxx!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fxx!)))
# fxu(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fxu)))
# fxu!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fxu!)))
# fuu(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fuu)))
# fuu!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(fuu!)))
# l(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(l)))
# l!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(l!)))
# lx(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lx)))
# lx!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lx!)))
# lu(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lu)))
# lu!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lu!)))
# lxx(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lxx)))
# lxx!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lxx!)))
# lxu(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lxu)))
# lxu!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(lxu!)))
# luu(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(luu)))
# luu!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(luu!)))
# p(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(p)))
# p!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(p!)))
# px(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(px)))
# px!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(px!)))
# pxx(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(pxx)))
# pxx!(M::Model,θ,t,ξ) = throw(ModelDefError(M,nameof(pxx!)))


# eg. Jx = Jacobian(x); fx_sym = Jx(f_sym)
# maps symbolic -> symbolic
struct Jacobian dv end

function (J::Jacobian)(f_sym)
    fv_sym = map(1:length(J.dv)) do i
        map(f_sym) do f
            derivative(f, J.dv[i])
        end
    end
    return cat(fv_sym...; dims=ndims(f_sym)+1) #ndims = 0 for scalar-valued l,p
end
# isnothing(force_dims) || (fx_sym = reshape(fx_sym, force_dims...))

macro model(T, ex)
    info("deriving a new model for $(as_bold(T))")

    # evaluate user-specified code sandboxed inside a module
    mod = eval(:(module $(gensym()) $ex end))
    iinfo("model evaluated to $mod\n")

    # extract dimensions
    NX = mod.NX; NU = mod.NU; NΘ = mod.NΘ

    # goal: build a set of expressions describing the user model to PRONTO
    M = Expr[]
    # define model type
    push!(M, striplines(:(struct $T <: PRONTO.Model{$NX,$NU,$NΘ} end)))
    push!(M, striplines(:(export $T))) # make it available in Main

    iinfo("initializing symbolics\n")
    # create symbolic variables & operators
    @variables x[1:NX] u[1:NU] θ[1:NΘ] t
    ξ = vcat(x,u)
    Jx,Ju = Jacobian.([x,u])

    # trace user expression with symbolic variables
    iinfo("tracing model\n")
    local f = collect(invokelatest(mod.f, collect(θ), t, collect(x), collect(u)))
    local l = collect(invokelatest(mod.l, collect(θ), t, collect(x), collect(u)))
    local p = collect(invokelatest(mod.p, collect(θ), t, collect(x), collect(u)))
    local Qr = collect(invokelatest(mod.Qr, collect(θ), t, collect(x), collect(u)))
    local Rr = collect(invokelatest(mod.Rr, collect(θ), t, collect(x), collect(u)))


    # generate method definitions for PRONTO functions
    iinfo("differentiating model dynamics\n")
    build_defs!(M, :f, T, (θ, t, ξ), f)
    build_defs!(M, :fx, T, (θ, t, ξ), f |> Jx)
    build_defs!(M, :fu, T, (θ, t, ξ), f |> Ju)
    build_defs!(M, :fxx, T, (θ, t, ξ), f |> Jx |> Jx)
    build_defs!(M, :fxu, T, (θ, t, ξ), f |> Jx |> Ju)
    build_defs!(M, :fuu, T, (θ, t, ξ), f |> Ju |> Ju)

    iinfo("differentiating stage cost\n")
    build_defs!(M, :l, T, (θ, t, ξ), l)
    build_defs!(M, :lx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX))
    build_defs!(M, :lu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU))
    build_defs!(M, :lxx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Jx)
    build_defs!(M, :lxu, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Ju)
    build_defs!(M, :luu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU) |> Ju)

    iinfo("differentiating terminal cost\n")
    build_defs!(M, :p, T, (θ, t, ξ), p)
    build_defs!(M, :px, T, (θ, t, ξ), p |> Jx |> px->reshape(px,NX))
    build_defs!(M, :pxx, T, (θ, t, ξ), p |> Jx |> px->reshape(px,NX) |> Jx)

    iinfo("building regulator functions\n")
    build_defs!(M, :Qr, T, (θ, t, ξ), Qr)
    build_defs!(M, :Rr, T, (θ, t, ξ), Rr)

    fname = tempname()*"_$T.jl"
    hdr = "#= this file was machine generated at $(now()) - DO NOT MODIFY =#\n\n"
    write(fname, hdr*prod(string.(M).*"\n\n"))
    info("model defined at $fname")
    quote
        include($fname)
        info("$(as_bold($T)) model available")
    end
end

# write("temp.jl", prod(string.(M).*"\n"))

build_defs!(M, name, T, args, sym) = append!(M, build_defs(name, T, args, sym))
build_defs(name, T, args, sym) = rename(build_function(sym, args...), name, T)


function rename(exps, name, T)
    defs = cleanup.(exps)
    defs[1].args[1] = :(PRONTO.$(name)(M::$T, $(defs[1].args[1].args...)))
    defs[2].args[1] = :(PRONTO.$(_!(name))(M::$T, $(defs[2].args[1].args...)))
    return defs
end

# makes generated definitions pretty
function cleanup(ex)
    # extract variable names from argument signature (for replacement)
    args = ex.args[1].args

    postwalk(striplines(ex)) do ex
        if isexpr(ex) && ex.head == :block && length(ex.args) == 1
            ex.args[1] # remove unused begin blocks
        # and rename matching variables:
        elseif length(args) > 3 && @capture(ex, $(args[end-3]))      
            :out
        elseif @capture(ex, $(args[end-2]))   
            :θ
        elseif @capture(ex, $(args[end-1]))   
            :t
        elseif @capture(ex, $(args[end]))     
            :ξ
        else 
            ex # otherwise, leave the expression unchanged
        end
    end
end

