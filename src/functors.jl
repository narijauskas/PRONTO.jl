# an object that allows non-allocating, type-stable returns

# B = Functor((B,t)->fu!(B,x(t),u(t)), NX, NU)
# B = Functor( (B,t)->fu!(B,xu(t)...), NX, NU)
# B = Functor(NX,NU) do B,t
#     fu!(B,xu(t)...)
# end
# struct Functor{F,S}
#     f!::F
#     X::MArray{S,Float64}
# end



#= object implementation
struct Functor{F,T}
    f!::F
    X::T
    Functor(f!::Function, X::MArray) = new(f!,X)
end

Functor(f!, dims...) = Functor(f!, buffer(dims...))
buffer(dims...) = MArray{Tuple{dims...},Float64}(undef)

(F::Functor)(args...) = (F.f!(F.X, args...); return F.X)

Base.show(io::IO, ::Functor{F,T}) where {F,T} = print(io, "Functor of $(T)")
=#

#TODO: goal is: what's the easiest way to combine

# B = buffer(NX,NU)
# fu!(B, x(t), u(t)) # or fu!(B, xu(t)...)
# return B
# should be zero-allocating & type stable



# Buffer{S} = MArray{S,Float64}
buffer(dims...) = MArray{Tuple{dims...},Float64}(undef)
functor(f!,X) = (F(args...) = (f!(X, args...); return X); return F)

# function functor(f!,dims...)
#     X = buffer(dims...)
#     function _F(args...)
#             f!(X, args...)
#         return X
#     return _F
# end

# A = functor((A,t)->fx!(A, x(t), u(t)), buffer(NX,NX)) # defines A(t)
# A = functor(buffer(NX,NX)) do A,t
#     fx!(A, x(t), u(t))
# end

# # zero allocations:
# F = functor((F,t) -> (F .= SVector{3}(1,2,3).*sin(t)), buffer(3)) # defines F(t)
# G = let F = F
#     functor((G,t) -> copy!(G, F(t)), buffer(3))
# end

# # alternatively, since let is clumsy
# function foo()
#     F = functor((F,t) -> (F .= SVector{3}(1,2,3).*sin(t)), buffer(3)) # defines F(t)
#     G = functor((G,t) -> copy!(G, F(t)), buffer(3))
#     return (F,G)
# end

# (F,G) = foo()
