using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60
benchmarks = Dict{String,Any}()

## ----------------------------------- ----------------------------------- ##

include("../examples/inv_pend.jl")
θ = InvPend() 
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
xf = @SVector [0;0]
u0 = @SVector [0.0]
α = t->xf
μ = t->u0
η = closed_loop(θ,x0,α,μ,τ)
benchmarks["InvPend_1"] = @benchmark pronto(θ, x0, η, τ; tol=1e-4, show_preview=false);

η = smooth(θ,x0,xf,τ) # worse initial guess
benchmarks["InvPend_2"] = @benchmark pronto(θ, x0, η, τ; tol=1e-4, show_preview=false);

## ----------------------------------- ----------------------------------- ##

include("../examples/two_spin.jl")
θ = TwoSpin() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [0.0, 1.0, 0.0, 0.0] # initial state
xf = @SVector [1.0, 0.0, 0.0, 0.0] # final state
μ = t->[0.1] # open loop input μ(t)
η = open_loop(θ, xf, μ, τ) # guess trajectory
benchmarks["TwoSpin"] = @benchmark pronto(θ, x0, η, τ; show_preview=false); # optimal trajectory

## ----------------------------------- ----------------------------------- ##

include("../examples/lane_change.jl")
θ = LaneChange(xeq = [1,0,0,0,0,0], kq=[0.1,0,1,0,0,0])
t0,tf = τ = (0,4)
x0 = SVector{6}(-5.0, zeros(5)...)
xf = @SVector zeros(6)
μ = t->zeros(2)
η = open_loop(θ,x0,μ,τ)
benchmarks["LaneChange"] = @benchmark pronto(θ, x0, η, τ; show_preview=false);

## ----------------------------------- ----------------------------------- ##




include("../examples/x_gate.jl")
θ = XGate3()
τ = t0,tf = 0,10
ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
benchmarks["XGate3"] = @benchmark pronto(θ, x0, η, τ; show_preview=false); # optimal trajectory







## ----------------------------------- ----------------------------------- ##

include("../examples/split.jl")
θ = Split2(kl=0.01, kr=1, kq=1)
t0,tf = τ = (0,10)
x0 = SVector{22}(x_eig(1))
μ = t->SVector{1}(0.4*sin(t))
η = open_loop(θ,x0,μ,τ)
benchmarks["Split2"] = @benchmark pronto(θ, x0, η, τ; show_preview=false); # optimal trajectory


