module PRONTO

using OrdinaryDiffEq
using ForwardDiff
using ForwardDiff: jacobian
using LinearAlgebra

include("trajectories.jl")
export Trajectory

include("implicit_diff.jl")
export Jx, Ju
export Hxx, Hxu, Huu

include("cost.jl")
include("pronto_main.jl")
include("riccati.jl")

end # module
