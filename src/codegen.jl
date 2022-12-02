# the tools for automatically building model derivatives


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



build_pretty(name, T, syms) = build_expr(name, T, syms; postprocess = prettify)

function build_expr(name, T, syms; postprocess = identity)
    # parallel map each symbolic to expr
    defs = tmap(enumerate(syms)) do (i,x)
        :(out[$i] = $(postprocess(toexpr(x))))
    end

    return quote
        function PRONTO.$name(θ::$T,x,u,t)
            out = SizedMatrix{$NX,$NX,Float64}(undef)
            @inbounds begin
                $(defs...)
            end
            return out
        end
    end |> clean
end


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


