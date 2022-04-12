module PRONTO
# __precompile__(false)


using LinearAlgebra
using Symbolics
using Symbolics: derivative
using DifferentialEquations
using DataInterpolations
# using ControlSystems # provides lqr
using MatrixEquations # provides arec

include("autodiff.jl")
# export jacobian
# export hessian


include("regulator.jl")
# export regulator

include("projection.jl")
# export project!, projection

end # module
