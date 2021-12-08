module PRONTO

using OrdinaryDiffEq
using Zygote


greet() = println("Hello World!")


include("ricatti.jl")

end # module
