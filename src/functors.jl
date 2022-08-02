
buffer(S) = MVector{S,Float64}(undef)
buffer(S1,S2) = MMatrix{S1,S2,Float64}(undef)

Buffer{S} = MArray{S,Float64}
Buffer{S}() where {S} = MArray{S,Float64}(undef)


functor(f!,X) = (F(args...) = (f!(X, args...); return X); return F)
# expanded, essentially:
# function functor(f!,X::Buffer)
#     function F(args...)
#             f!(X, args...)
#         return X
#     return F
# end
