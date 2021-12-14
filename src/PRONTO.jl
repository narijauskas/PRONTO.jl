module PRONTO

using OrdinaryDiffEq
using ForwardDiff
using ForwardDiff: jacobian
using LinearAlgebra
using MatrixEquations

include("trajectories.jl")
export Trajectory

include("implicit_diff.jl")
export Jx, Ju
export Hxx, Hxu, Huu

# include("cost.jl")
# include("riccati.jl")
# include("pronto_main.jl")

end # module
