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

include("cost.jl")
export build_LQ_cost
include("riccati.jl")
export optKr
include("pronto_main.jl")
export project, pronto

end # module
