struct Bar{T}
    x::T
    y::T
end

@variables x
@variables y

bar = Bar(x,y)

using Symbolics
using Symbolics: scalarize
Symbolics.variables(:x, 1:4)

x = (@variables x[1:4]; return x)
scalarize(x)