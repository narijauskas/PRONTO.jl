# the tools for automatically building model derivatives

using Base: invokelatest
using MacroTools: striplines, prettify, postwalk, @capture

# change this to MacroTools.prettify if you want pretty code and are willing to wait for it
global format::Function = identity


# ----------------------------------- code generation macros ----------------------------------- #

macro define_f(T, ex)
    names = fieldnames(eval(:(Main.$T)))
    fn = :((x,u,t, $(names...)) -> $ex)
    :(define_f($(esc(T)), $(esc(fn))))
end

macro define_l(T, ex)
    names = fieldnames(eval(:(Main.$T)))
    fn = :((x,u,t, $(names...)) -> $ex)
    :(define_l($(esc(T)), $(esc(fn))))
end

macro define_m(T, ex)
    names = fieldnames(eval(:(Main.$T)))
    fn = :((x,u,t, $(names...)) -> $ex)
    :(define_m($(esc(T)), $(esc(fn))))
end

macro define_Qr(T, ex)
    names = fieldnames(eval(:(Main.$T)))
    fn = :((x,u,t, $(names...)) -> $ex)
    :(define_Qr($(esc(T)), $(esc(fn))))
end

macro define_Rr(T, ex)
    names = fieldnames(eval(:(Main.$T)))
    fn = :((x,u,t, $(names...)) -> $ex)
    :(define_Rr($(esc(T)), $(esc(fn))))
end

# friendly names for the macros
var"@dynamics" = var"@define_f"
var"@incremental_cost" = var"@define_l"
var"@terminal_cost" = var"@define_m"
var"@regulator_Q" = var"@define_Qr"
var"@regulator_R" = var"@define_Rr"

# run before using model, ensures all methods are generated
function resolve_model(T::Type{<:Model})
    define_L(T)
    #MAYBE: verify existence of all kernel methods
    info(PRONTO.as_bold(T)*" model ready")
    return nothing
end

# ----------------------------------- code generation pipeline ----------------------------------- #

# builds buffered versions by default, for an in-place version, use:
struct InPlace end # TODO: deprecate?

function define_f(T::Type{<:Model{NX,NU}}, user_f) where {NX,NU}
    info("defining dynamics and derivatives for $(as_bold(T))")
    f = traceuser(T, user_f)
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
    l = traceuser(T, user_l)
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

function define_m(T::Type{<:Model{NX,NU}}, user_m) where {NX,NU}
    info("defining terminal cost and derivatives for $(as_bold(T))")
    m = traceuser(T, user_m)
    Jx,Ju = jacobians(T)
    mx = reshape(Jx(m), NX)
    define_methods(T, Size(1), m, :m, :x, :u, :t)
    define_methods(T, Size(NX), mx, :mx, :x, :u, :t)
    define_methods(T, Size(NX,NX), Jx(mx), :mxx, :x, :u, :t)
    return nothing
end

function define_Qr(T::Type{<:Model{NX,NU}}, user_Q) where {NX,NU}
    info("defining regulator Q method for $(as_bold(T))")
    Q = traceuser(T, user_Q)
    define_methods(T, Size(NX,NX), Q, :Q, :x, :u, :t)
    return nothing
end

function define_Rr(T::Type{<:Model{NX,NU}}, user_R) where {NX,NU}
    info("defining regulator R method for $(as_bold(T))")
    R = traceuser(T, user_R)
    define_methods(T, Size(NU,NU), R, :R, :x, :u, :t)
    return nothing
end

function define_L(T::Type{<:Model{NX,NU}}) where {NX,NU}
    info("defining Lagrangian methods for $(as_bold(T))")
    Jx,Ju = jacobians(T)

    f = trace(T, PRONTO.f)
    fxx = Jx(Jx(f))
    fxu = Ju(Jx(f))
    fuu = Ju(Ju(f))

    l = trace(T, PRONTO.l)
    lx = reshape(Jx(l), NX)
    lu = reshape(Ju(l), NU)
    lxx = Jx(lx)
    lxu = Ju(lx)
    luu = Ju(lu)

    λ = collect(first(@variables λ[1:NX]))
    Lxx = lxx .+ sum(λ[k]*fxx[k,:,:] for k in 1:NX)
    Lxu = lxu .+ sum(λ[k]*fxu[k,:,:] for k in 1:NX)
    Luu = luu .+ sum(λ[k]*fuu[k,:,:] for k in 1:NX)

    define_methods(T, Size(NX,NX), Lxx, :Lxx, :λ, :x, :u, :t)
    define_methods(T, Size(NX,NU), Lxu, :Lxu, :λ, :x, :u, :t)
    define_methods(T, Size(NU,NU), Luu, :Luu, :λ, :x, :u, :t)
    return nothing
end


# ----------------------------------- code generation helpers ----------------------------------- #

# object to facilitate symbolic differentiation of model
struct SymModel{T} end

# symmodel.kq -> variables fitting 
getproperty(::SymModel{T}, name::Symbol) where {T<:Model} = symbolic(name, fieldtype(T, name))
propertynames(::SymModel{T}) where T = fieldnames(T)
Base.all(M::SymModel{T}) where T = [getproperty(M, name) for name in propertynames(M)]

# for models
symbolic(T::Type{<:Model}) = SymModel{T}()

# scalar variables and fields (default)
symbolic(name::Symbol, ::Type{<:Any}) = symbolic(name)
symbolic(name::Symbol) = first(@variables $name)

# array variables and fields
symbolic(name::Symbol, T::Type{<:StaticArray}) = symbolic(name, Size(T))
symbolic(name::Symbol, ::Size{S}) where S = collect(first(@variables $name[[1:N for N in S]...]))
symbolic(name::Symbol, S::Vararg{Int}) = collect(first(@variables $name[[1:N for N in S]...]))

# non-static arrays
symbolic(name::Symbol, T::Type{<:AbstractArray}) = error("Cannot create symbolic representation of variable-sized array. Consider using StaticArrays.")

# for pretty versions of symbols
symbolic(name::Symbol, ix::UnitRange) = Symbolics.variables(name, ix)

function traceuser(T::Type{<:Model{NX,NU}}, fn) where {NX,NU}
    x = symbolic(:x, NX)
    u = symbolic(:u, NU)
    t = symbolic(:t)
    return collect(fn(x, u, t, all(symbolic(T))...))
end

function trace(T::Type{<:Model{NX,NU}}, f) where {NX,NU}
    x = symbolic(:x, NX)
    u = symbolic(:u, NU)
    t = symbolic(:t)
    θ = symbolic(T)
    # NOTE: invokelatest might not be necessary
    return collect(invokelatest(f, θ, x, u, t))
end

function jacobians(T::Type{<:Model{NX,NU}}) where {NX,NU}
    x = collect(first(@variables x[1:NX]))
    u = collect(first(@variables u[1:NU]))
    Jx,Ju = Jacobian.([x,u])
    return Jx,Ju
end

# functor which maps symbolic -> symbolic jacobian
# eg. Jx = Jacobian(x); fx_sym = Jx(f_sym)
struct Jacobian dv end

function (J::Jacobian)(f_sym)
    fv_sym = map(1:length(J.dv)) do i
        map(f_sym) do f
            derivative(f, J.dv[i])
        end
    end
    return cat(fv_sym...; dims=ndims(f_sym)+1) #ndims = 0 for scalar-valued l,p
end

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
    hdr = "PRONTO.$name(θ::$(T)$([", $a" for a in args]...))" |> x->x*repeat(" ", max(36-length(x),0))
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
        function PRONTO.$name!(out, θ::Union{$T,SymModel{$T}}, $(args...))
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
        function PRONTO.$name(θ::SymModel{$T}, $(args...))
            out = Array{Num}(undef, $(S...))
            PRONTO.$name!(out, θ, $(args...))
            return SArray{Tuple{$(S...)}, Num}(out)
        end
    end |> clean
end

function def_generic(name, T, sz::Size{S}, args...) where S
    name! = _!(name)
    return quote
        function PRONTO.$name(θ::$T, $(args...))
            out = $(MType(sz))(undef)
            PRONTO.$name!(out, θ, $(args...))
            return $(SType(sz))(out)
        end
    end |> clean
end


# append ! to a symbol, eg. :name -> :name!
_!(ex) = Symbol(String(ex)*"!")

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
