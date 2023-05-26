# the tools for automatically building model derivatives

using Base: invokelatest
using MacroTools: striplines, prettify, postwalk, @capture

global format::Function = identity

# builds buffered versions by default, for an in-place version, use:
struct InPlace end

# export generate_model, InPlace
# export build
# symbolic(name::Symbol, indices...) = Symbolics.variables(name::Symbol, indices...)
# symbolic(T::Type{<:Model}) = SymbolicModel(T)
# export symbolic

macro dynamics(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(define_f($(esc(T)), $(esc(fn))))
end

macro stage_cost(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(define_l($(esc(T)), $(esc(fn))))
end

macro terminal_cost(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(define_p($(esc(T)), $(esc(fn))))
end

macro regulatorQ(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(define_Q($(esc(T)), $(esc(fn))))
end

macro regulatorR(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(define_R($(esc(T)), $(esc(fn))))
end

macro lagrangian(T)
    :(define_L($(esc(T))))
end

#TODO: resolve_model

export @dynamics, @stage_cost, @terminal_cost, @regulatorQ, @regulatorR, resolve_model


function resolve_model(T::Type{<:Model})
    #TODO: verify existence of all kernel methods
    define_L(T)
    info(PRONTO.as_bold(T)*" model ready")
    return nothing
end



function define_f(T::Type{<:Model{NX,NU}}, user_f) where {NX,NU}
    info("defining dynamics and derivatives for $(as_bold(T))")
    f = trace(T, user_f)
    Jx,Ju = jacobians(T)
    define_methods(T, Size(NX), f, :f, :x, :u, :t)
    define_methods(T, Size(NX,NX), Jx(f), :fx, :x, :u, :t)
    define_methods(T, Size(NX,NU), Ju(f), :fu, :x, :u, :t)
    define_methods(T, Size(NX,NX,NX), Jx(Jx(f)), :fxx, :x, :u, :t)
    define_methods(T, Size(NX,NX,NU), Ju(Jx(f)), :fxu, :x, :u, :t)
    define_methods(T, Size(NX,NU,NU), Ju(Ju(f)), :fuu, :x, :u, :t)
    return nothing
end

function define_l(T::Type{<:Model{NX,NU}}, user_l) where {NX,NU}
    info("defining stage cost and derivatives for $(as_bold(T))")
    l = trace(T, user_l)
    Jx,Ju = jacobians(T)
    lx = reshape(Jx(l), NX)
    lu = reshape(Ju(l), NU)
    define_methods(T, Size(1), l, :l, :x, :u, :t)
    define_methods(T, Size(NX), lx, :lx, :x, :u, :t)
    define_methods(T, Size(NU), lu, :lu, :x, :u, :t)
    define_methods(T, Size(NX,NX), Jx(lx), :lxx, :x, :u, :t)
    define_methods(T, Size(NX,NU), Ju(lx), :lxu, :x, :u, :t)
    define_methods(T, Size(NU,NU), Ju(lu), :luu, :x, :u, :t)
    return nothing
end

function define_p(T::Type{<:Model{NX,NU}}, user_p) where {NX,NU}
    info("defining terminal cost and derivatives for $(as_bold(T))")
    p = trace(T, user_p)
    Jx,Ju = jacobians(T)
    px = reshape(Jx(p), NX)
    define_methods(T, Size(1), p, :p, :x, :u, :t)
    define_methods(T, Size(NX), px, :px, :x, :u, :t)
    define_methods(T, Size(NX,NX), Jx(px), :pxx, :x, :u, :t)
    return nothing
end

function define_Q(T::Type{<:Model{NX,NU}}, user_Q) where {NX,NU}
    info("defining regulator Q method for $(as_bold(T))")
    Q = trace(T, user_Q)
    define_methods(T, Size(NX,NX), Q, :Q, :x, :u, :t)
    return nothing
end

function define_R(T::Type{<:Model{NX,NU}}, user_R) where {NX,NU}
    info("defining regulator R method for $(as_bold(T))")
    R = trace(T, user_R)
    define_methods(T, Size(NU,NU), R, :R, :x, :u, :t)
    return nothing
end

function define_L(T::Type{<:Model{NX,NU}}) where {NX,NU}
    info("defining lagrangian methods for $(as_bold(T))")

    fxx = trace(T, PRONTO.fxx)
    fxu = trace(T, PRONTO.fxu)
    fuu = trace(T, PRONTO.fuu)

    lxx = trace(T, PRONTO.lxx)
    lxu = trace(T, PRONTO.lxu)
    luu = trace(T, PRONTO.luu)

    λ = collect(first(@variables λ[1:NX]))
    Lxx = lxx .+ sum(λ[k]*fxx[k,:,:] for k in 1:NX)
    Lxu = lxu .+ sum(λ[k]*fxu[k,:,:] for k in 1:NX)
    Luu = luu .+ sum(λ[k]*fuu[k,:,:] for k in 1:NX)

    define_methods(T, Size(NX,NX), Lxx, :Lxx, :λ, :x, :u, :t)
    define_methods(T, Size(NX,NU), Lxu, :Lxu, :λ, :x, :u, :t)
    define_methods(T, Size(NU,NU), Luu, :Luu, :λ, :x, :u, :t)
    return nothing
end


# #FUTURE: options to make pretty (or add other postprocessing), save to file, etc.
# function build_f(T, user_f)
#     info("building dynamics methods for $(as_bold(T))")
#     NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

#     f = invokelatest(user_f, x, u, t, θ)

#     build(Size(NX), :(f(x,u,t,θ::$M)), f, M)

#     build(Size(NX,NX), :(fx(x,u,t,θ::$M)), Jx(f), M)
#     build(Size(NX,NU), :(fu(x,u,t,θ::$M)), Ju(f), M)

#     build(Size(NX,NX,NX), :(fxx(x,u,t,θ::$M)), Jx(Jx(f)), M)
#     build(Size(NX,NX,NU), :(fxu(x,u,t,θ::$M)), Ju(Jx(f)), M)
#     build(Size(NX,NU,NU), :(fuu(x,u,t,θ::$M)), Ju(Ju(f)), M)
    
#     return nothing
# end

# function build_l(T, user_l)
#     info("building stage cost methods for $(as_bold(T))")
#     NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

#     l = invokelatest(user_l, x, u, t, θ)
#     lx = reshape(Jx(l), NX)
#     lu = reshape(Ju(l), NU)

#     build(Size(1), :(l(x,u,t,θ::$M)), l, M)
#     build(Size(NX), :(lx(x,u,t,θ::$M)), lx, M)
#     build(Size(NU), :(lu(x,u,t,θ::$M)), lu, M)

#     build(Size(NX,NX), :(lxx(x,u,t,θ::$M)), Jx(lx), M)
#     build(Size(NX,NU), :(lxu(x,u,t,θ::$M)), Ju(lx), M)
#     build(Size(NU,NU), :(luu(x,u,t,θ::$M)), Ju(lu), M)
#     return nothing
# end

# function build_p(T, user_p)
#     info("building terminal cost methods for $(as_bold(T))")
#     NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

#     p = invokelatest(user_p, x, u, t, θ)
#     px = reshape(Jx(p), NX)

#     build(Size(1), :(p(x,u,t,θ::$M)), p, M)
#     build(Size(NX), :(px(x,u,t,θ::$M)), px, M)
#     build(Size(NX,NX), :(pxx(x,u,t,θ::$M)), Jx(px), M)
#     return nothing
# end

# function build_Q(T, user_Q)
#     info("building regulator Q method for $(as_bold(T))")
#     NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

#     Q = invokelatest(user_Q, x, u, t, θ)
#     build(Size(NX,NX), :(Q(x,u,t,θ::$M)), Q, M)
#     return nothing
# end

# function build_R(T, user_R)
#     info("building regulator R method for $(as_bold(T))")
#     NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

#     R = invokelatest(user_R, x, u, t, θ)
#     build(Size(NU,NU), :(R(x,u,t,θ::$M)), R, M)
#     return nothing
# end

# trace(f, vars...) = invokelatest(f, vars...)
# fxx = symbolic(T, PRONTO.fxx)
# trace(f, model, x, u, t)




# θ = symbolic(T)
function trace(T::Type{<:Model{NX,NU}}, f) where {NX,NU}
    x = collect(first(@variables x[1:NX]))
    u = collect(first(@variables u[1:NU]))
    t = first(@variables t)
    θ = symbolic(T)
    return collect(invokelatest(f, x, u, t, θ))
    # return Symbolics.scalarize(invokelatest(f, x, u, t, θ))
    # return collect(invokelatest(f, x, u, t, θ))
end

function jacobians(T::Type{<:Model{NX,NU}}) where {NX,NU}
    x = collect(first(@variables x[1:NX]))
    u = collect(first(@variables u[1:NU]))
    Jx,Ju = Jacobian.([x,u])
    return Jx,Ju
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






function define_methods(T, sz::Size{S}, sym, name, args...) where S
    body = def_kernel(sym)
    file = tempname()*".jl"
    write(
        file, 
        def_generic(name, T, sz, args...) |> string, "\n\n",
        def_symbolic(name, T, sz, args...) |> string, "\n\n",
        def_inplace(name, T, body, args...) |> string, "\n\n",
    )
    Base.include(Main, file)
    hdr = "PRONTO.$name($(["$a, " for a in args]...)θ::$T)" |> x->x*repeat(" ", max(36-length(x),0))
    iinfo("$hdr "*as_color(crayon"dark_gray", "[$file]"))
end


function def_kernel(sym)
    kernel = tmap(enumerate(sym)) do (i,x)
        :(out[$i] = $(format(toexpr(x))))
    end
    # line below fixes discrepancy between map and tmap for zero-dim arrays
    return kernel isa AbstractArray ? kernel : [kernel]
end

function def_inplace(name, T, body, args...)
    name! = _!(name)
    ex =  quote
        function PRONTO.$name!(out, $(args...), θ::Union{$T,SymModel{$T}})
            @inbounds begin
                $(body...)
            end
            return out
        end
    end |> clean

    # re-reference model fields (eg. kq->θ.kq)
    for name in fieldnames(T)
        name in (:x,:u,:t) && error("model cannot have fields named x,u,t")
        ex = crispr(ex, name, :(θ.$name))
    end

    return ex
end

function def_symbolic(name, T, sz::Size{S}, args...) where S
    name! = _!(name)
    return quote
        function PRONTO.$name($(args...), θ::SymModel{$T})
            out = Array{Num}(undef, $(S...))
            PRONTO.$name!(out, $(args...), θ)
            return SArray{Tuple{$(S...)}, Num}(out)
        end
    end |> clean
end

function def_generic(name, T, sz::Size{S}, args...) where S
    name! = _!(name)
    return quote
        function PRONTO.$name($(args...), θ::$T)
            out = $(MType(sz))(undef)
            PRONTO.$name!(out, $(args...), θ)
            return $(SType(sz))(out)
        end
    end |> clean
end


# append ! to a symbol, eg. :name -> :name!
_!(ex) = Symbol(String(ex)*"!")


# AType(sz::Size{S}) where {S} = :(Array{T}(undef, $(S...)))

# generate efficient constructors for numeric SArrays and MArrays
MType(sz::Size{S}) where {S} = MType(Val(length(S)), sz)
MType(::Val{1}, sz::Size{S}) where {S} = :(MVector{$(S...), Float64})
MType(::Val{2}, sz::Size{S}) where {S} = :(MMatrix{$(S...), Float64})
MType(::Val, sz::Size{S}) where {S} = :(MArray{Tuple{$(S...)}, Float64, $(length(S)), $(prod(S))})

SType(sz::Size{S}) where {S} = SType(Val(length(S)), sz)
SType(::Val{1}, sz::Size{S}) where {S} = :(SVector{$(S...), Float64})
SType(::Val{2}, sz::Size{S}) where {S} = :(SMatrix{$(S...), Float64})
SType(::Val, sz::Size{S}) where {S} = :(SArray{Tuple{$(S...)}, Float64, $(length(S)), $(prod(S))})






# make a version of v with the sparsity pattern of fn
function sparse_mask(v, fn)
    v .* map(fn) do ex
        iszero(ex) ? 0 : 1
    end |> collect
end


# remove excess begin blocks & comments
function clean(ex)
    postwalk(striplines(ex)) do ex
        isexpr(ex) && ex.head == :block && length(ex.args) == 1 ? ex.args[1] : ex
    end
end


# insert the `new` expression at each matching `tgt` in the `src`
function crispr(src,tgt,new)
    postwalk(src) do ex
        return @capture(ex, $tgt) ? new : ex
    end
end





























#MAYBE: do we actually want to save the whole model to a file?
#TODO: deprecate
function build(sz, hdr, sym, M; file=nothing)
    body = tmaparr(enumerate(sym)) do (i,x)
        :(out[$i] = $(format(toexpr(x))))
    end
    @capture(hdr, name_(args__))
    file = tempname()*".jl"
    ex = _build(sz, name, args[1:end-1], body, M)
    write(file, (@. ex |> clean |> x->string(x)*"\n\n")...)
    Base.include(Main, file)
    iinfo("$hdr\t"*as_color(crayon"dark_gray", "[$file]"))
    # eval(ex)
end

# separate(str) = str*"\n\n"
# :(θ::$M{T})
#TODO: deprecate
function _build(sz::Size{S}, name, args, body, M) where {S}
    name! = _!(name)
    f1 = quote
        function PRONTO.$name($(args...), θ::$M{T}) where {T<:Number}
            out = $(MType(sz))(undef)
            PRONTO.$name!(out, $(args...), θ)
            # @inbounds begin
            #     $(body...)
            # end
            return $(SType(sz))(out)
        end
    end

    f2 = quote
        function PRONTO.$name($(args...), θ::$M{Num})
            out = Array{Num}(undef, $(S...))
            PRONTO.$name!(out, $(args...), θ)
            return SArray{Tuple{$(S...)}}(out)
        end
    end

    # f3 = quote
    #     function PRONTO.$name($(args...), θ::$M{T}) where {T}
    #         out = Array{T}(undef, $(S...))
    #         PRONTO.$name!(out, $(args...), θ)
    #         return SArray{Tuple{$(S...)}}(out)
    #     end
    # end

    f4 = quote
        function PRONTO.$name!(out, $(args...), θ::$M)
            @inbounds begin
                $(body...)
            end
            return out
        end
    end

    return (f1,f2,f4)
end

# function _build(::InPlace, name, args, body, M)
#     quote
#         function PRONTO.$name($(args...)) where {T}
#             @inbounds begin
#                 $(body...)
#             end
#             return nothing
#         end
#     end
# end

#TODO: deprecate
function init_syms(T)
    NX = nx(T)
    NU = nu(T)
    @variables x[1:NX] u[1:NU] t λ[1:NX]
    x = collect(x)
    u = collect(u)
    λ = collect(λ)
    t = t
    θ = SymbolicModel(T)
    Jx,Ju = Jacobian.([x,u])
    # M = :($(nameof(T)){T})
    M = nameof(T)
    return (NX,NU,x,u,t,θ,λ,Jx,Ju,M)
end







# fix discrepancy between map and tmap for zero-dim arrays
# TODO: deprecate
function tmaparr(f, args...) 
    x = tmap(f, args...)
    return x isa AbstractArray ? x : [x]
end

# build_inplace(:f, body, :(model::TwoSpin), :x, :u, :t)
