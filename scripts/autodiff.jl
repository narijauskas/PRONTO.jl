# a quick demo of automatic differentiation

using ForwardDiff
using ForwardDiff: derivative, gradient
using BenchmarkTools

fn(t) = cos(t) + cos(t)
df(t) = derivative(fn, t)
df_manual(t) = -sin(t) - sin(t)

T = 0.0

@code_typed fn(T)
@code_typed df(T)

@code_native fn(T)
@code_native df(T)

@btime fn(T)
@btime df(T)
@btime df_manual(T)

##
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, xs, f.(xs))
lines!(ax, xs, df.(xs))
display(fig)