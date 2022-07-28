
# What's an better way to achieve:
# A = buffer(...)
# ...
# fx!(A, x(t), u(t))

# should be zero-allocating & type stable

buffer(dims...) = MArray{Tuple{dims...},Float64}(undef)
functor(f!,X) = (F(args...) = (f!(X, args...); return X); return F)

export buffer, functor
# function functor(f!,dims...)
#     X = MArray{Tuple{dims...},Float64}(undef)
#     function F(args...)
#             f!(X, args...)
#         return X
#     return F
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


