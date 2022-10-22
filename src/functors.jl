using StaticArrays
# buffer(S) = MVector{S,Float64}(undef)
# buffer(S1,S2) = MMatrix{S1,S2,Float64}(undef)
buffer(S...) = MArray{Tuple{S...},Float64}(undef)

Buffer{S} = MArray{S,Float64}
Buffer{S}() where {S} = MArray{S,Float64}(undef)

macro buffer(S...)
    return :(Buffer{Tuple{$(S...)}}())
end

# function functor(f!,x,u,dims...)
#     A = buffer(dims...)
#     return @closure (t)->(f!(A,x(t),u(t)); return A)
# end

functor(f!,X) = (F(args...) = (f!(X, args...); return X); return F)
# expanded, essentially:
# function functor(f!,X::Buffer)
#     function F(args...)
#             f!(X, args...)
#         return X
#     return F
# end



# struct Functor{S,F,NX,NU}
#     f!::F
#     ξ::Trajectory{NX,NU}
#     buf::MArray{S,Float64}
# end

# (A::Functor)(x,u) = (A.f!(A.buf, x, u); return A.buf)
# (A::Functor)(t) = A(A.x(t), A.u(t))

# function Functor{S}(f!::F,ξ) where {S,F<:Function}
#     buf = Buffer{S}()
#     Functor{S,F}(f!, ξ, buf)
# end