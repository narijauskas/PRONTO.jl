module PRONTO
# __precompile__(false)


using LinearAlgebra
using Symbolics
using Symbolics: derivative
using DifferentialEquations
using DataInterpolations
# using ControlSystems # provides lqr
using MatrixEquations # provides arec

# helper functions
tau(f, t) = LinearInterpolation(hcat(map(f, t)...), t)



include("autodiff.jl")
# export jacobian
# export hessian


include("regulator.jl")
# export regulator

include("projection.jl")
# export project!, projection

include("cost.jl")

include("gradient_descent.jl")

end # module
