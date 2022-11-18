using MacroTools
using MacroTools: postwalk, striplines
using Base: invokelatest
using Symbolics
using SparseArrays
using Dates

# function dim(N, ex)
#     v = 0
#     postwalk(exi->(@capture(exi, $N=val_) && (v = val); exi) , ex)
#     return v
# end


# # key must use `val_` somewhere, eg key = :(NX = val_)
# function extract(ex, key)
#     v = nothing
#     postwalk(exi->(@capture(exi, $key) && (v = val); exi) , ex)
#     isnothing(v) && @error "no match found"
#     return v
# end



# function build(f::Function, args...)
#     f_sym = Base.invokelatest(f, args...) # trace with symbolic args
#     # symbolic processing would go here, eg. fxu_sym = Ju(Jx(f))
#     f_ex = build_function(collect(f_sym), args...) # symbolic -> expression
#     return f_ex
# end

# assuming f(θ,t,ξ)
# function build(f::Function)
#     @variables x[1:NX] u[1:NU] θ[1:NΘ] t
#     ξ = vcat(x,u)

#     f_sym = Base.invokelatest(f, θ, t, ξ) # trace with symbolic args
#     # symbolic processing would go here, eg. fxu_sym = Ju(Jx(f_sym))
#     f_ex = build_function(collect(f_sym), θ, t, ξ) # symbolic -> expression
#     return f_ex
# end

#TODO: jacobian Jx maps f_sym to fx_sym with force reshape option
# sym2ex(Jx(Ju(ex2sym(ex))))
# fxu_ex, fxu!_ex = ex |> ex2sym |> Jx |> Ju |> sym2ex
# struct Jacobian dv end
# (J::Jacobian)(f_sym) = apply_jacobian(J.dv, f_sym)


# eg. Jx = Jacobian(x); fx_sym = Jx(f_sym)
# maps symbolic -> symbolic
function f end
function f! end
function fx end
function fx! end
function fu end
function fu! end
function fxx end
function fxx! end
function fxu end
function fxu! end
function fuu end
function fuu! end
function l end
function l! end
function lx end
function lx! end
function lu end
function lu! end
function lxx end
function lxx! end
function lxu end
function lxu! end
function luu end
function luu! end

struct Jacobian
    dv
end

# (J::Jacobian)(f_sym) = J([f_sym])
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
    # evaluate user-specified code sandboxed inside a module
    mod = eval(:(module $(gensym()) $ex end))
    info("code expanded to $mod")

    # extract dimensions
    NX = mod.NX; NU = mod.NU; NΘ = mod.NΘ

    # goal: build a set of expressions describing the user model to PRONTO
    M = Expr[]
    # define model type
    push!(M, striplines(:(struct $T <: PRONTO.Model{$NX,$NU,$NΘ} end)))
    push!(M, striplines(:(export $T)))

    # create symbolic variables & operators
    @variables x[1:NX] u[1:NU] θ[1:NΘ] t
    ξ = vcat(x,u)
    Jx,Ju = Jacobian.([x,u])

    # trace user expression with symbolic variables
    local f = invokelatest(mod.f, collect(θ), t, collect(x), collect(u))
    local l = invokelatest(mod.l, collect(θ), t, collect(x), collect(u))

    # generate method definitions for PRONTO functions
    build_defs!(M, :f, T, (θ, t, ξ), f)
    build_defs!(M, :fx, T, (θ, t, ξ), f |> Jx)
    build_defs!(M, :fu, T, (θ, t, ξ), f |> Ju)
    build_defs!(M, :fxx, T, (θ, t, ξ), f |> Jx |> Jx)
    build_defs!(M, :fxu, T, (θ, t, ξ), f |> Jx |> Ju)
    build_defs!(M, :fuu, T, (θ, t, ξ), f |> Ju |> Ju)

    build_defs!(M, :l, T, (θ, t, ξ), l)
    build_defs!(M, :lx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX))
    build_defs!(M, :lu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU))
    build_defs!(M, :lxx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Jx)
    build_defs!(M, :lxu, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Ju)
    build_defs!(M, :luu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU) |> Ju)


    fname = tempname()*"_$T.jl"
    hdr = "#= this file was machine generated at $(now()) - DO NOT MODIFY =#\n"
    write(fname, hdr*prod(string.(M).*"\n\n"))
    info("model written to $fname")
    quote
        include($fname)
    end
end

# write("temp.jl", prod(string.(M).*"\n"))

build_defs!(M, name, T, args, sym) = append!(M, build_defs(name, T, args, sym))
build_defs(name, T, args, sym) = rename(build_function(sym, args...), name, T)

_!(ex) = Symbol(String(ex)*"!")

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


    # sym = Base.invokelatest(f, args...) # trace with symbolic args
    # symbolic processing would go here, eg. fxu_sym = Ju(Jx(f))
    # f_ex = build_function(collect(f_sym), args...) # symbolic -> expression
    # f,f! = build_function(collect(sym), args...) # symbolic -> expression
    # push!(M, postprocess!(f, name, T))

    # f_sym = trace(f, args...)
    # trace user expression with symbolic variables, and convert from
    # f_sym = invokela(θ,t,ξ) do θ,t,ξ
    #     x,u = (ξ[1:NX], ξ[(NX+1):end])
    # end
    

#     T = :Placeholder
#     M = Expr[]

#     build_function(f, θ, t, ξ)
#     postprocess
#     add header
#     push!(M)

#     append!(M, build(T, :f, f_sym))
#     define!(M, T, :fx, f_sym |> Jx)
#     define!(M, T, :fu, f_sym |> Ju)

#     # trace
#     # process symbolic
#     # build_function
#     # process expression
#     # push to

#     # convert user f(θ,t,x,u) to f(θ,t,ξ)
#     local f_ex,f!_ex = build(θ,t,ξ) do θ,t,ξ
#         x,u = (ξ[1:NX], ξ[(NX+1):end])
#         mod.f(θ,t,x,u)
#     end
#     return f!_ex
# end

# function trace(f, args...)
#     Base.invokelatest(f, args...) # trace with symbolic args
# end

# function define!(M, T, name, sym)
#     foreach(build_function(collect(sym), args...)) do ex # symbolic -> expression
#         push!(M, postprocess(ex, name, T))
#     end
# end

# function postprocess(ex)
#     args = ex.args[1].args

#     postwalk(striplines(ex)) do ex
#         if length(args)>5 && @capture(ex, $(args[end-3]))
#             :buf
#         elseif @capture(ex, $(args[end-2]))
#             :θ
#         elseif @capture(ex, $(args[end-1]))
#             :t
#         elseif @capture(ex, $(args[end]))
#             :ξ
#         else
#             ex
#         end
#     end
# end
#=

# derivatives:
# base
# fx
 
#NOTE: The model/derivatives assume that:
# the user-defined l(...) and p(...) are scalar-valued (ie, ndims(p(...)) == 0),
# whereas f(...) is vector-valued (ie, ndims(f(...)) == 1).

macro model(name, ex)

    # extract dimensions from user code (might not be needed)
    NX = dim(:NX, ex)
    NU = dim(:NU, ex)
    NΘ = dim(:NΘ, ex)

    # run user-specified code inside of a temporary module
    mod = gensym()
    eval(:(module $mod $ex end))

    # defines local f(x)
    # eval(ex)


    # convert f(θ,t,x,u) to f(θ,t,ξ)
    # what if we do this last?


    # check for user functions
    #FUTURE: use @capture on a postwalk of the expression to check for correct form
    # @isdefined f && f isa Function || @error "could not find a definition for the function f"

    # make symbolic variables
    @variables x[1:NX] u[1:NU] θ[1:NΘ] t
    ξ = vcat(x,u)

    # sometimes want to specify modifications at the anonymous level, eg. below,
    # other times want to specify operators for the symbolic level, eg. Jx
    # convert user f(θ,t,x,u) to f(θ,t,ξ)
    local f,f! = build(θ,t,ξ) do θ,t,ξ
        x,u = split(ξ)
        mod.f(θ,t,x,u)
    end
    # call the user function with symbolic args
    f_sym = collect(Base.invokelatest(f,θ,t,x,u))
    # process the symbolic expression
    # either do nothing, or apply Jx and/or Ju
    # should be able to chain for hessianx

    # make ξ refactored x_ex

    # local fxu,fxu! = Ju(fx,θ,t,ξ)
    # local fuu,fuu! = Ju(fu,θ,t,ξ)

    # call w./ symbolics, define local functions as expressions
    f_ex = build_function(fx_sym, args...)

    f_ex.args[1] = :(PRONTO.f(M::$name,θ,t,ξ))
    # f_sym = f(θ,t,x,u)

    # for each, 

    quote

        struct $name <: PRONTO.Model{$NX,$NU,$NΘ} end
        
    end
end

# goal: assign PRONTO.f and PRONTO.f!


# replace f(θ,t,x,u) with 


# can we assume everything is f(θ,)
# assume ex is a function (args...) body end 
# function yeet(ex)::Expr
#     eval(ex)
#     # todo, first populate symbolics
#     # every call should simply be (θ,t,ξ)
#     f_sym = collect(Base.invokelatest(f, args...))
#     f_ex,f!_ex = build_function(f_sym,θ,t,x,u)

# end



function yeet(ex)::Expr
    NX = dim(:NX, ex)
    NU = dim(:NU, ex)
    NΘ = dim(:NΘ, ex)

    @variables θ[1:NΘ]
    @variables t
    @variables x[1:NX] 
    @variables u[1:NU]

    # postwalk(ex->(@capture(ex,$N=v_) && (y=v); ex) , nex)

    if @capture(ex, f_(args__) = body_)
        # f_sym = collect(Base.invokelatest(eval(ex), args...)) # args are symbolics
        f_sym = collect(Base.invokelatest(eval(ex), θ, t, ξ)) # should be common order
        f_ex, f!_ex = build_function(f_sym, args...)
    end
    # if @capture(ex, f(θ,t,x,u) = user_f_)
    # could modify user function, eg. collect() all symbolics?

    # return quot
    #     PRONTO.f(M::$T,θ,t,ξ) = f(θ,t,ξ) # NX
    #     PRONTO.f!(M::$T,buf,θ,t,ξ) = f!(buf,θ,t,ξ) # NX
    # end
    return f_ex
end

# maybe the move is to generate the f_ex def with renamed PRONTO method header
# and the instruction to eval(f_ex)


    # load function definition expression
    # evaluate & symbolically derive
    # add definitions to pronto


function _symbolics(T)::Expr
    return quote

        # create symbolic variables
        @variables θ[1:nθ($T())]
        @variables t
        @variables x[1:nx($T())] 
        @variables u[1:nu($T())]
        ξ = vcat(x,u)
        @variables α[1:nx($T())] 
        @variables μ[1:nu($T())]
        φ = vcat(α,μ)
        @variables z[1:nx($T())] 
        @variables v[1:nu($T())]
        ζ = vcat(z,v)
        @variables α̂[1:nx($T())] 
        @variables μ̂[1:nu($T())]
        φ̂ = vcat(α̂,μ̂)
        @variables Pr[1:nx($T()),1:nx($T())]
        @variables Po[1:nx($T()),1:nx($T())]
        @variables ro[1:nx($T())]
        @variables λ[1:nx($T())]
        @variables γ
        @variables y[1:2] #YO: can we separate these into scalar Dh/D2g?
        @variables h #MAYBE: rename j or J?

        # create Jacobian operators
        Jx,Ju = Jacobian.([x,u])
    end
end
=#