module PRONTO
# __precompile__(false)

# using ForwardDiff
# using ForwardDiff: jacobian, gradient, hessian
using LinearAlgebra
using MatrixEquations
# using DifferentialEquations

include("autodiff.jl")
# export jacobian


# include("trajectories.jl")
# export Trajectory

# include("implicit_diff.jl")
# export Jx, Ju
# export Hxx, Hxu, Huu

# include("cost.jl")
# export build_LQ_cost


# include("pronto_main.jl")
# export optKr
# export project, pronto

# include("search_direction.jl")
# export search_direction

end # module
