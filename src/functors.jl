# an object that allows non-allocating, type-stable returns

# Ar = Functor(model.fx, X_α, U_μ, NX, NX)
# Ar(t) updates in-place and returns SArray{T}

# Interpolation{S,T} = LinearInterpolation{Vector{SArray{S,T}}, Vector{Float64}}

# S must be Tuple{dims...}
struct Functor1{FT,S}
    buf::MArray{S,Float64}
    fxn::FT
end

# Kr!(src, invRr, Br, Pr) = ... # inplace version!
# Kr = Functor(Kr!, dims, invRr, Br, Pr)


function Functor1(fxn!, dims, args...)
    buf = MArray{Tuple{dims...},Float64}(undef) #FIX: generalize T beyond F64?
    fxn = t->fxn!(buf, (arg(t) for arg in args)...)
    # Functor1{typeof(fxn), Tuple{dims...}}(buf,fxn)
    Functor1(buf, fxn)
end

# update!(A::Functor1, t) = A.fxn(t) # update buffer
function(A::Functor1{S})(t) where {S}
    # update!(A,t)
    A.fxn(t)
    return SArray(A.buf)::SArray{S,Float64} # enforce type? ::MArray{S,Float64}
end

# Functor{T}
struct Functor3{FT,S1,S2,T}
    buf::MMatrix{S1,S2,T}
    fxn::FT
    X # ::Interpolation
    U # ::Interpolation

    function Functor3(fxn::FT, X, U, dims...) where {FT}
        buf = MMatrix{dims..., Float64}(undef) #FIX: generalize properly
        new{FT,dims...,Float64}(buf,fxn,X,U)
    end
end


#YO: for a 100x speedup:
# model.fx! = jacobian(x,f,x,u; inplace=true)
# model.fx!(A.buf, A.X(t), A.U(t))

function update!(A::Functor3, t)
    # in-place update to buf
    # A.buf .= A.fxn(A.X(t), A.U(t))
    A.buf .= _update!(A.fxn, A.X(t), A.U(t)) 
    # map!(t->(), A.buf, 1.0)
end

_update!(fxn,x,u) = fxn(x,u)
# _update!(A.fxn, A.X(t), A.U(t))