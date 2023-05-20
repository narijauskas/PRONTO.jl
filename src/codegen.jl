# the tools for automatically building model derivatives

using Base: invokelatest
using MacroTools: striplines, prettify, postwalk, @capture

global format::Function = identity

# builds buffered versions by default, for an in-place version, use:
struct InPlace end

# export generate_model, InPlace
# export build
symbolic(name::Symbol, indices...) = Symbolics.variables(name::Symbol, indices...)
symbolic(T::Type{<:Model}) = SymbolicModel(T)
export symbolic

macro dynamics(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(build_f($(esc(T)), $(esc(fn))))
end

macro stage_cost(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(build_l($(esc(T)), $(esc(fn))))
end

macro terminal_cost(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(build_p($(esc(T)), $(esc(fn))))
end

macro regulatorQ(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(build_Q($(esc(T)), $(esc(fn))))
end

macro regulatorR(T, ex)
    fn = :((x,u,t,θ)->$ex)
    :(build_R($(esc(T)), $(esc(fn))))
end

macro lagrangian(T)
    :(build_L($(esc(T))))
end

export @dynamics, @stage_cost, @terminal_cost, @regulatorQ, @regulatorR ,@lagrangian



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

#FUTURE: options to make pretty (or add other postprocessing), save to file, etc.
function build_f(T, user_f)
    info("building dynamics methods for $(as_bold(T))")
    NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

    f = invokelatest(user_f, x, u, t, θ)

    build(Size(NX), :(f(x,u,t,θ::$M)), f, M)

    build(Size(NX,NX), :(fx(x,u,t,θ::$M)), Jx(f), M)
    build(Size(NX,NU), :(fu(x,u,t,θ::$M)), Ju(f), M)

    build(Size(NX,NX,NX), :(fxx(x,u,t,θ::$M)), Jx(Jx(f)), M)
    build(Size(NX,NX,NU), :(fxu(x,u,t,θ::$M)), Ju(Jx(f)), M)
    build(Size(NX,NU,NU), :(fuu(x,u,t,θ::$M)), Ju(Ju(f)), M)
    
    return nothing
end

function build_l(T, user_l)
    info("building stage cost methods for $(as_bold(T))")
    NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

    l = invokelatest(user_l, x, u, t, θ)
    lx = reshape(Jx(l), NX)
    lu = reshape(Ju(l), NU)

    build(Size(1), :(l(x,u,t,θ::$M)), l, M)
    build(Size(NX), :(lx(x,u,t,θ::$M)), lx, M)
    build(Size(NU), :(lu(x,u,t,θ::$M)), lu, M)

    build(Size(NX,NX), :(lxx(x,u,t,θ::$M)), Jx(lx), M)
    build(Size(NX,NU), :(lxu(x,u,t,θ::$M)), Ju(lx), M)
    build(Size(NU,NU), :(luu(x,u,t,θ::$M)), Ju(lu), M)
    return nothing
end

function build_p(T, user_p)
    info("building terminal cost methods for $(as_bold(T))")
    NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

    p = invokelatest(user_p, x, u, t, θ)
    px = reshape(Jx(p), NX)

    build(Size(1), :(p(x,u,t,θ::$M)), p, M)
    build(Size(NX), :(px(x,u,t,θ::$M)), px, M)
    build(Size(NX,NX), :(pxx(x,u,t,θ::$M)), Jx(px), M)
    return nothing
end

function build_Q(T, user_Q)
    info("building regulator Q method for $(as_bold(T))")
    NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

    Q = invokelatest(user_Q, x, u, t, θ)
    build(Size(NX,NX), :(Q(x,u,t,θ::$M)), Q, M)
    return nothing
end

function build_R(T, user_R)
    info("building regulator R method for $(as_bold(T))")
    NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

    R = invokelatest(user_R, x, u, t, θ)
    build(Size(NU,NU), :(R(x,u,t,θ::$M)), R, M)
    return nothing
end

function build_L(T)
    info("building lagrangian methods for $(as_bold(T))")
    NX,NU,x,u,t,θ,λ,Jx,Ju,M = init_syms(T)

    _fxx = invokelatest(fxx,x,u,t,θ)
    _fxu = invokelatest(fxu,x,u,t,θ)
    _fuu = invokelatest(fuu,x,u,t,θ)

    _lxx = invokelatest(lxx,x,u,t,θ)
    _lxu = invokelatest(lxu,x,u,t,θ)
    _luu = invokelatest(luu,x,u,t,θ)

    Lxx = _lxx .+ sum(λ[k]*_fxx[k,:,:] for k in 1:NX)
    Lxu = _lxu .+ sum(λ[k]*_fxu[k,:,:] for k in 1:NX)
    Luu = _luu .+ sum(λ[k]*_fuu[k,:,:] for k in 1:NX)

    build(Size(NX,NX), :(Lxx(λ,x,u,t,θ::$M)), Lxx, M)
    build(Size(NX,NU), :(Lxu(λ,x,u,t,θ::$M)), Lxu, M)
    build(Size(NU,NU), :(Luu(λ,x,u,t,θ::$M)), Luu, M)
    return nothing
end

# append ! to a symbol, eg. :name -> :name!
_!(ex) = Symbol(String(ex)*"!")


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






# Array{T}(undef, S...)

AType(sz::Size{S}) where {S} = :(Array{T}(undef, $(S...)))

MType(sz::Size{S}) where {S} = MType(Val(length(S)), sz)
MType(::Val{1}, sz::Size{S}) where {S} = :(MVector{$(S...), T})
MType(::Val{2}, sz::Size{S}) where {S} = :(MMatrix{$(S...), T})
MType(::Val, sz::Size{S}) where {S} = :(MArray{Tuple{$(S...)}, T, $(length(S)), $(prod(S))})

SType(sz::Size{S}) where {S} = SType(Val(length(S)), sz)
SType(::Val{1}, sz::Size{S}) where {S} = :(SVector{$(S...), T})
SType(::Val{2}, sz::Size{S}) where {S} = :(SMatrix{$(S...), T})
SType(::Val, sz::Size{S}) where {S} = :(SArray{Tuple{$(S...)}, T, $(length(S)), $(prod(S))})




# fix discrepancy between map and tmap for zero-dim arrays
function tmaparr(f, args...) 
    x = tmap(f, args...)
    return x isa AbstractArray ? x : [x]
end

#MAYBE: do we actually want to save the whole model to a file?
function build(sz, hdr, sym, M; file=nothing)
    body = tmaparr(enumerate(sym)) do (i,x)
        :(out[$i] = $(format(toexpr(x))))
    end
    @capture(hdr, name_(args__))
    ex = _build(sz, name, args[1:end-1], body, M)
    file = tempname()*".jl"
    write(file, (@. ex |> clean |> string |> separate)...)
    Base.include(Main, file)
    iinfo("$hdr\t"*as_color(crayon"dark_gray", "[$file]"))
    # eval(ex)
end

separate(str) = str*"\n\n"
# :(θ::$M{T})
function _build(sz::Size{S}, name, args, body, M) where {S}
    name! = _!(name)
    f1 = quote
        function PRONTO.$name($(args...), θ::$M{T}) where {T<:Number}
            out = $(MType(sz))(undef)
            PRONTO.$name!(out, $(args...), θ)
            return $(SType(sz))(out)
        end
    end

    f2 = quote
        function PRONTO.$name($(args...), θ::$M{Num})
            out = Array{Num}(undef, $(S...))
            PRONTO.$name!(out, $(args...), θ)
            return SArray{Tuple{$(S...)}}(out)
            # return $(SType(sz))(out)
        end
    end

    f3 = quote
        function PRONTO.$name($(args...), θ::$M{T}) where {T}
            out = Array{T}(undef, $(S...))
            PRONTO.$name!(out, $(args...), θ)
            return SArray{Tuple{$(S...)}}(out)
        end
    end

    f4 = quote
        function PRONTO.$name!(out, $(args...), θ::$M)
            @inbounds begin
                $(body...)
            end
            return out
        end
    end

    return (f1,f2,f3,f4)
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

















#=
#FUTURE: options to make pretty (or add other postprocessing), save to file, etc.
function generate_model(T, user_f, user_l, user_p, user_Q, user_R)
    info("generating the $(as_bold(T)) model")
    iinfo("initializing symbolics...")
    NX = nx(T); NU = nu(T)
    @variables x[1:NX] u[1:NU] t λ[1:NX]
    x = collect(x)
    u = collect(u)
    t = t
    θ = SymbolicModel(T)
    λ = collect(λ)

    Jx,Ju = Jacobian.([x,u])

    iinfo("tracing functions for $T...")
    f = invokelatest(user_f, x, u, t, θ)
    l = invokelatest(user_l, x, u, t, θ)
    p = invokelatest(user_p, x, u, t, θ)
    Q = invokelatest(user_Q, x, u, t, θ)
    R = invokelatest(user_R, x, u, t, θ)

    build(InPlace(), :(f!(out,x,u,t,θ::$T)), f)

    build(Size(NX,NX), :(Q(x,u,t,θ::$T)), Q)
    build(Size(NU,NU), :(R(x,u,t,θ::$T)), R)

    fx = Jx(f)
    fu = Ju(f)

    build(Size(NX), :(f(x,u,t,θ::$T)), f)
    build(Size(NX,NX), :(fx(x,u,t,θ::$T)), fx)
    build(Size(NX,NU), :(fu(x,u,t,θ::$T)), fu)

    lx = reshape(Jx(l),NX)
    lu = reshape(Ju(l),NU)

    build(Size(1), :(l(x,u,t,θ::$T)), l)
    build(Size(NX), :(lx(x,u,t,θ::$T)), lx)
    build(Size(NU), :(lu(x,u,t,θ::$T)), lu)

    lxx = Jx(lx)
    lxu = Ju(lx)
    luu = Ju(lu)

    build(Size(NX,NX), :(lxx(x,u,t,θ::$T)), lxx)
    build(Size(NX,NU), :(lxu(x,u,t,θ::$T)), lxu)
    build(Size(NU,NU), :(luu(x,u,t,θ::$T)), luu)

    fxx = Jx(Jx(f))
    fxu = Ju(Jx(f))
    fuu = Ju(Ju(f))

    #YO: can reload defs from inplace versions
    Lxx = lxx .+ sum(λ[k]*fxx[k,:,:] for k in 1:NX)
    Lxu = lxu .+ sum(λ[k]*fxu[k,:,:] for k in 1:NX)
    Luu = luu .+ sum(λ[k]*fuu[k,:,:] for k in 1:NX)

    build(Size(NX,NX), :(Lxx(λ,x,u,t,θ::$T)), Lxx)
    build(Size(NX,NU), :(Lxu(λ,x,u,t,θ::$T)), Lxu)
    build(Size(NU,NU), :(Luu(λ,x,u,t,θ::$T)), Luu)

    px = reshape(Jx(p),NX)

    build(Size(1), :(p(x,u,t,θ::$T)), p)
    build(Size(NX), :(px(x,u,t,θ::$T)), px)
    build(Size(NX,NX), :(pxx(x,u,t,θ::$T)), Jx(px))
    iinfo("done!")
    nothing
end
=#