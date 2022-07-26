# include("../test/test_spin.jl")

## -------------------------------------- setup -------------------------------------- ##

using PRONTO
using LinearAlgebra
using Symbolics
using StaticArrays
using BenchmarkTools
using JET

NX = 4
NU = 1
ts = 0:0.001:10
T = last(ts)
x0 = MVector{NX, Float64}(0,1,0,0)
u0 = MVector{NU, Float64}(0)

model = (
    NX = NX, NU = NU, ts = ts,
    x0 = x0,
    tol = 1e-5,
    x_eq = [1.0;0.0;0.0;0.0], #For this example, we don't have any equilibrium points... xf is the target state here 
    u_eq = zeros(NU),
    maxiters = 20,
    α = 0.4,
    β = 0.7,
)

#NOTE: This is a really cool property of anonymous functions
# foo = ((x::T) where {T})->(T)
# foo(1)
# foo(1.0)

H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]

# f(x,u) = (H0 + u*H1)*x;
# fx(x,u) = H0 + u*H1;
# fu(x,u) = H1*x;

f = (x,u) -> collect((H0 + u[1]*H1)*x)
fx_manual = (x,u)->(H0 + u[1]*H1)


Ql = zeros(NX,NX)
Rl = 0.01
Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1] #terminal cost matrix Pl

l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
p = (x)-> 1/2*collect(x)'*Pl*collect(x)

@info "running autodiff"
model = autodiff(model,f,l,p)
@info "autodiff complete"


fx_auto = model.fx
fx_auto! = model.fx!

fx_model = (x,u)->(model.fx(x,u))
fx_model! = (buf,x,u)->(model.fx!(buf,x,u))

fx_pass! = (buf,x,u,fx!)->(fx!(buf,x,u))
fx_pass2! = (buf,x,u,model)->(model.fx!(buf,x,u))
fx_pass3! = (buf,x,u,model)->begin
    fx! = model.fx!
    fx!(buf,x,u)
end
fx_pass4! = (buf,x,u,model)->begin
    fx! = model.fx!
    fx_pass!(buf,x,u,fx!)
end

## -------------------------------------- benchmarks -------------------------------------- ##
# FX buffer
isdefined(Main, :FX) && const FX = MMatrix{NX,NX,Float64}(undef)


# baseline
@benchmark fill!(FX, 0) # 2.7 ns
@benchmark fill!(FX, rand()) # 7.2 ns

# regular versions
@benchmark fx_manual(x0,u0) # 254 ns
@benchmark fx_auto(x0,u0) # 2.7 μs
@benchmark fx_model(x0,u0) # 2.8 μs


# in-place versions
@benchmark fx_auto!(FX,x0,u0) # 23 ns, zero allocs
@benchmark fx_model!(FX,x0,u0) # 185 ns
@benchmark fx_pass!(FX,x0,u0,model.fx!) # 166 ns
@benchmark fx_pass2!(FX,x0,u0,model) # 38 ns, zero allocs
@benchmark fx_pass3!(FX,x0,u0,model) # 42 ns, zero allocs
@benchmark fx_pass4!(FX,x0,u0,model) # 64 ns, zero allocs


# conclusions
# ~10x speedup from in-place autodiff
# ~10x speedup when zero allocs
# zero allocs occur if implemented by pulling fx! from model (bizzare)