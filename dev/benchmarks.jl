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

fx_functor = Functor(NX,NX) do buf,x,u
    fx_auto!(buf,x,u)
end

fx_functor2 = Functor(NX,NX) do buf,x,u
    model.fx!(buf,x,u)
end

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
fx_pass5! = (buf,x,u,model)->begin
    fx!(buf,x,u) = model.fx!(buf,x,u)
    fx!(buf,x,u)
end

fx_test! = (buf,α,μ,t,model)->begin
    fx!(buf,x,u) = model.fx!(buf,x,u)
    fx!(buf,α(t),μ(t))
end

fx_test2! = (buf,α,μ,t,model)->begin
    fx! = copy(model.fx!)
    fx!(buf,α(t),μ(t))
end
fx_test3! = (buf,α,μ,t,fx!)->begin
    fx!(buf,α(t),μ(t))
end

fx_test4! = (buf,α,μ,t;fx!)->begin
    fx!(buf,α(t),μ(t))
end
## -------------------------------------- benchmarks -------------------------------------- ##
# FX buffer
isdefined(Main, :FX) || const FX = MMatrix{NX,NX,Float64}(undef)


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
@benchmark fx_pass5!(FX,x0,u0,model) # 38 ns, zero allocs
@report_opt fx_pass5!(FX,x0,u0,model) # clean

@benchmark fx_functor(x0,u0) # 369 ns
@benchmark fx_functor2(x0,u0) # 475 ns

# conclusions
# ~10x speedup from in-place autodiff
# ~10x speedup when zero allocs
# zero allocs occur if implemented by pulling fx! from model (bizzare)
# functors don't seem to solve the problem

## -------------------------------------- interpolants -------------------------------------- ##
# X buffer
isdefined(Main, :X) || const X = MVector{NX,Float64}(undef)
isdefined(Main, :X2) || const X2 = MVector{NX,Float64}(undef)


α = Interpolant(ts, NX)
μ = Interpolant(ts, NU)

t = 3.5
@benchmark α(t) # 161 ns
@benchmark μ(t) # 121 ns

@benchmark fill!(X, 0) # 2.7 ns
@benchmark fill!(X, rand()) # 6.8 ns
@benchmark copy!(X2, X) # 2.5 ns

@benchmark copy!(X, α(t)) # 188 ns
@benchmark X.=α(t) # 1.7 μs

@benchmark fx_model!(FX, α(t), μ(t)) # 468 ns
@benchmark fx_auto!(FX, α(t), μ(t)) # 298 ns

## -------------------------------------- type stability -------------------------------------- ##

@code_warntype α(t) # type stable
@report_opt α(t) # no issues

@btime fx_auto!(FX,x0,u0) # 28 ns
@code_warntype fx_auto!(FX,x0,u0) # type stable
@report_opt fx_auto!(FX,x0,u0) # no issues
@allocated fx_auto!(FX,x0,u0) # 0

@btime fx_auto!(FX, α(t), μ(t)) # 301 ns
@code_warntype fx_auto!(FX, α(t), μ(t)) # type stable
@report_opt fx_auto!(FX, α(t), μ(t)) # no issues
@allocated fx_auto!(FX, α(t), μ(t)) # 96

@benchmark fx_test!(FX,α,μ,t,model) # 273 ns
@code_warntype fx_test!(FX,α,μ,t,model) # type stable
@report_opt fx_test!(FX,α,μ,t,model) # clean
@allocated fx_test!(FX,α,μ,t,model) # 96

@benchmark fx_test3!(FX,α,μ,t,fx_auto!) # 256 ns
@code_warntype fx_test3!(FX,α,μ,t,fx_auto!) # type stable

@benchmark fx_test3!(FX,α,μ,t,model.fx!) # 804 ns
@code_warntype fx_test3!(FX,α,μ,t,model.fx!) # type stable

@benchmark fx_test4!(FX,α,μ,t; fx! = fx_auto!) # 547 ns
@benchmark fx_test4!(FX,α,μ,t; fx! = model.fx!) # 668 ns


#TODO: make zero-allocating interpolant
#TODO: pass separate functions instead of combined PRONTO model
