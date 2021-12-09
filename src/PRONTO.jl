module PRONTO
include("trajectories.jl")
include("cost.jl")
include("pronto_main.jl")


using OrdinaryDiffEq
using ForwardDiff

greet() = println("Hello World!")


include("ricatti.jl")

end # module
