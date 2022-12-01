using MacroTools
using MacroTools: postwalk, striplines
using Base: invokelatest
using Symbolics
using SparseArrays
using Dates

# or just throw a method error?
struct ModelDefError <: Exception
    M::Model
    fn::Symbol
end

function Base.showerror(io::IO, e::ModelDefError)
    T = typeof(e.M)
    print(io, "PRONTO.$(e.fn) is missing a method for the $T model.\n")
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

# @define f    (θ,t,ξ)
# @define fx   (θ,t,ξ)
# @define fu   (θ,t,ξ)
# @define fxx  (θ,t,ξ)
# @define fxu  (θ,t,ξ)
# @define fuu  (θ,t,ξ)
# @define l    (θ,t,ξ)
# @define lx   (θ,t,ξ)
# @define lu   (θ,t,ξ)
# @define lxx  (θ,t,ξ)
# @define lxu  (θ,t,ξ)
# @define luu  (θ,t,ξ)
# @define p    (θ,t,ξ) 
# @define px   (θ,t,ξ)
# @define pxx  (θ,t,ξ)
# @define Qrr   (θ,t,ξ)
# @define Rrr   (θ,t,ξ)

#TODO: don't dispatch on model
Ar!(out,θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Ar!)))
Ar(θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Ar)))
Br!(out,θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Br!)))
Br(θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Br)))
# Kr!(out,θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Kr!)))
Kr(θ::Model,t,ξ,Pr) = Rr(θ,t,ξ)\(Br(θ,t,ξ)'Pr)
Qr!(out,θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Qr!)))
Qr(θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Qr)))
Rr!(out,θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Rr!)))
Rr(θ::Model,t,ξ) = throw(ModelDefError(M,nameof(Rr)))

f!(dx,θ,ξ,t) = @error "no f! defined"
f(θ,ξ,t) = @error "no f defined"

views(::Model{NX,NU,NΘ},ξ) where {NX,NU,NΘ} = (@view ξ[1:NX]),(@view ξ[NX+1:end])

function forced_dynamics!(dξ,ξ,(θ,g),t)
    _,u = views(θ,ξ)
    dx,du = views(θ,dξ)

    f!(dx,θ,ξ,t)
    copyto!(du, g(t) - u)
    return nothing
end

function regulated_dynamics!(dξ,ξ,(θ,φ,Pr),t)
    x,u = views(θ,ξ)
    dx,du = views(θ,dξ)
    α,μ = views(θ,φ(t))

    f!(dx,θ,ξ,t)
    copyto!(du, μ - Kr(θ,t,ξ,Pr)*(x-α) - u)
    return nothing
end




closed_loop!(dξ,ξ,(θ,φ,Pr),t) = _closed_loop!(dξ,ξ,(θ,φ,Pr),t)
function _closed_loop!(dξ,ξ,(θ,φ,Pr),t)
    # x,u = views(θ,ξ)
    dx,du = views(θ,dξ)
    # α,μ = views(θ,φ(t))

    f!(dx,θ,ξ,t)
    # copyto!(du, μ - Kr(θ,t,φ,Pr)*(x-α) - u)
    copyto!(du, Δu(θ,ξ,φ(t),Pr,t))
    return nothing
end

Δu(θ,ξ,φ,Pr,t) = Δu(θ,views(θ,ξ)...,views(θ,φ)...,Pr,t)
Δu(θ,x,u,α,μ,Pr,t) = μ - Kr(θ,t,[α;μ],Pr)*(x-α) - u



function dynamics!(dx,x,(θ,φ,Pr),t)
    α,μ = views(θ,φ(t))
    u = μ - Kr(θ,t,φ(t),Pr)*(x-α)
    ξ = [x;u]
    f!(dx,θ,ξ,t)
end


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
# struct Yeet2  <: Model{22,1,0}
#     Ar::MMatrix{NX,NX,Float64,NX*NX}
#     Br::MMatrix{NX,NU,Float64,NX*NU}
# end

# function Yeet2(#=future θ=#)
#     Yeet2(MMatrix())
# end

# $build_buf(:Ar, NX, NX)
function buffer_def(name, dims...)
    :($name::MArray{Tuple{$(dims...)},Float64,$(length(dims)),$(prod(dims))})
    # if 0 == length(dims)
    #     :($name::MArray{Tuple{$(dims...)},Float64,$(length(dims))})
    # else
    # end
end
function buffer_init(name, dims...)
    :(zeros(MArray{Tuple{$(dims...)},Float64}))
end
function type_def(T,NX,NU,NΘ)
    # buffers = Dict
    ex = quote
        export $T
        struct $T <: PRONTO.Model{$NX,$NU,$NΘ}
            $(buffer_def(:θ, NΘ))
            $(buffer_def(:Ar, NX, NX))
            $(buffer_def(:Br, NX, NU))
            $(buffer_def(:Qr, NX, NX))
            $(buffer_def(:Rr, NU, NU))
            $(buffer_def(:Kr, NU, NX))
        end
        function $T()
            $T(
                $(buffer_init(:θ, NΘ)),
                $(buffer_init(:Ar, NX, NX)),
                $(buffer_init(:Br, NX, NU)),
                $(buffer_init(:Qr, NX, NX)),
                $(buffer_init(:Rr, NU, NU)),
                $(buffer_init(:Kr, NU, NX))
            )
        end
    end
    striplines(ex).args
end




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

function model(T, ex)
    info("deriving a new model for $(as_bold(T))")

    # evaluate user-specified code sandboxed inside a module
    mdl = eval(:(module $(gensym()) $ex end))
    iinfo("model evaluated to $mdl\n")

    # extract dimensions
    NX = mdl.NX; NU = mdl.NU; NΘ = mdl.NΘ

    # goal: build a set of expressions describing the user model to PRONTO
    M = Expr[]
    # define model type
    # push!(M, striplines(:(struct $T <: PRONTO.Model{$NX,$NU,$NΘ} end)))
    # push!(M, striplines(:(export $T))) # make it available in Main
    append!(M, type_def(T,NX,NU,NΘ))

    iinfo("initializing symbolics\n")
    # create symbolic variables & operators
    @variables x[1:NX] u[1:NU] θ[1:NΘ] t
    ξ = vcat(x,u)
    Jx,Ju = Jacobian.([x,u])

    # trace user expression with symbolic variables
    iinfo("tracing model\n")
    local f = collect(invokelatest(mdl.f, collect(θ), t, collect(x), collect(u)))
    local l = collect(invokelatest(mdl.l, collect(θ), t, collect(x), collect(u)))
    local p = collect(invokelatest(mdl.p, collect(θ), t, collect(x), collect(u)))
    local Qr = collect(invokelatest(mdl.Qr, collect(θ), t, collect(x), collect(u)))
    local Rr = collect(invokelatest(mdl.Rr, collect(θ), t, collect(x), collect(u)))
    return Jx,Ju,f,l,p,Qr,Rr

    # # generate method definitions for PRONTO functions
    # iinfo("differentiating model dynamics\n")
    # build_defs!(M, :f, T, (θ, t, ξ), f)
    # build_defs!(M, :fx, T, (θ, t, ξ), Jx(f))
    # build_defs!(M, :fu, T, (θ, t, ξ), Ju(f))
    # build_defs!(M, :fxx, T, (θ, t, ξ), Jx(Jx(f)))
    # build_defs!(M, :fxu, T, (θ, t, ξ), Ju(Jx(f)))
    # build_defs!(M, :fuu, T, (θ, t, ξ), Ju(Ju(f)))

    # iinfo("differentiating stage cost\n")
    # build_defs!(M, :l, T, (θ, t, ξ), l)
    # build_defs!(M, :lx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX))
    # build_defs!(M, :lu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU))
    # build_defs!(M, :lxx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Jx)
    # build_defs!(M, :lxu, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Ju)
    # build_defs!(M, :luu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU) |> Ju)
    # # @build lx(θ,t,ξ)->reshape(l|>Jx,NX)

    # iinfo("differentiating terminal cost\n")
    # build_defs!(M, :p, T, (θ, t, ξ), p)
    # build_defs!(M, :px, T, (θ, t, ξ), p |> Jx |> px->reshape(px,NX))
    # build_defs!(M, :pxx, T, (θ, t, ξ), p |> Jx |> px->reshape(px,NX) |> Jx)

    # iinfo("building regulator functions\n")
    # build_defs!(M, :Qrr, T, (θ, t, ξ), Qr)
    # build_defs!(M, :Rrr, T, (θ, t, ξ), Rr)



    # @variables 
    # local B = f |> Ju
    # local Kr = collect(Rr\collect(B'*Pr))
    # cleanup(collect(build_function(Kr, θ, t, ξ, Pr; parallel=Symbolics.MultithreadedForm(),expression=Val{true}))[2], :θ, :t, :ξ, :Pr)
    # build_defs!(M, :Kr, T, (θ, t, ξ, Pr), collect(Rr\collect(B'*Pr)))
    
    return M
end


function build_dP()
    @variables Pr[1:NX,1:NX]
    # @variables Kr[1:NU,1:NX]
    @variables Rr[1:NU,1:NU]
    @variables Qr[1:NX,1:NX]
    @variables Ar[1:NX,1:NX]
    @variables Br[1:NX,1:NU]

    Ar = sparse_mask(Ar, Jx(f))
    Kr = collect(Rr \ collect(Br'*Pr))
    # fx_mask
end


# make a version of v with the sparsity pattern of fn
function sparse_mask(v, fn)
    v .* map(fn) do ex
        iszero(ex) ? 0 : 1
    end |> collect
end


macro model(T, ex)
    mdl = gensym()
    eval(:(module $mdl $ex end))
    iinfo("model evaluated to temporary module: PRONTO.var\"$mdl\"\n")
    return mdl
    # ignored:
    
    # M = model(T,ex)

    # fname = tempname()*"_$T.jl"
    # hdr = "#= this file was machine generated at $(now()) - DO NOT MODIFY =#\n\n"
    # write(fname, hdr*prod(string.(M).*"\n\n"))
    # info("model defined at $fname")
    # quote
    #     include($fname)
    #     info("$(as_bold($T)) model available")
    # end
end

# write("temp.jl", prod(string.(M).*"\n"))
#=
build_defs!(M, name, T, args, sym) = append!(M, build_defs(name, T, args, sym))
build_defs(name, T, args, sym) = rename(build_function(sym, args...), name, T)


function rename(exps, name, T)
    defs = cleanup.(exps)
    defs[1].args[1] = :(PRONTO.$(name)(M::$T, $(defs[1].args[1].args...)))
    defs[2].args[1] = :(PRONTO.$(_!(name))(M::$T, $(defs[2].args[1].args...)))
    return defs
end

# remove excess begin blocks
function unwrap(ex)
    postwalk(striplines(ex)) do ex
        isexpr(ex) && ex.head == :block && length(ex.args) == 1 ? ex.args[1] : ex
    end
end



function cleanup(ex, args...)
    # extract generated variable names from argument signature (for replacement)
    vars = ex.args[1].args
    # arg_names = collect(Symbolics.getname.(args))
    arg_names = collect(args)
    length(vars) == length(arg_names) + 1 && pushfirst!(arg_names, :out)
    postwalk(striplines(ex)) do ex
        # remove unused begin blocks
        if isexpr(ex) && ex.head == :block && length(ex.args) == 1
            return ex.args[1]
        end
        # and rename matching variables:
        for (i,name) in enumerate(arg_names)
            if @capture(ex, $(vars[i]))
                return name
            end
        end
        # otherwise, leave the expression unchanged
        return ex
    end
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

=#
















function build_struct(T,NX,NU,NΘ)
    :(
        struct $T <: Model{$NX,$NU,$NΘ}
            #TODO: parameter support
        end
    )
end




# remove excess begin blocks & comments
function strip(ex)
    postwalk(striplines(ex)) do ex
        isexpr(ex) && ex.head == :block && length(ex.args) == 1 ? ex.args[1] : ex
    end
end


# macro build(ex)
#     @capture(ex, name_{dims__}(args__) = def_)
#     @show eval.(esc.(args))
#     :()
#     # strip(build_function(eval(def), eval.(args)...)[2])
# end

define(symex, symargs...)::Expr = strip(build_function(symex, symargs...)[2])

# build_inplace()

# sort of redundant
body(args, symex, symargs...) = rename_args(define(symex, symargs...),args)

# rename each oldargs with each args
function rename_args(def,args)::Expr
    oldargs = def.args[1].args
    newargs = vcat(:out, args)
    postwalk(def) do ex
        for (i,argname) in enumerate(newargs)
            if @capture(ex, $(oldargs[i]))
                return argname
            end
        end
        # otherwise, leave the expression unchanged
        return ex
    end.args[2]
end

# generate method definitions to add to PRONTO
function build_methods(name,T,dims,args,def)
    striplines(quote
        function PRONTO.$(_!(name))(out, $(θ_dispatch(T,args...)...))
            $(rename_args(def,args))
        end

        function PRONTO.$name($(θ_dispatch(T,args...)...))
            out = SizedArray{Tuple{$(dims...)},Float64}(undef)
            PRONTO.$(_!(name))(out, $(args...))
            return out
        end
    end).args
end

function θ_dispatch(T,args...)
    map(args) do ex
        ex == :θ ? :(θ::$T) : ex
    end
end

# insert the `new` expression at each matching `tgt` in the `src`
function crispr(src,tgt,new)
    postwalk(src) do ex
        return @capture(ex, tgt) ? new : ex
    end
end


# defs[1].args[1] = :(PRONTO.$(name)(M::$T, $(defs[1].args[1].args...)))
# defs[2].args[1] = :(PRONTO.$(_!(name))(M::$T, $(defs[2].args[1].args...)))

# PRONTO.Br(θ::Split,)




# build sparsity-aware custom version
# @variables A[1:NX,1:NX] -> match sparsity pattern of fx
# generate kernel Kr, dPr_dt


# benchmark compare
# also compare against multithreaded