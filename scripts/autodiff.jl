using ForwardDiff
using ForwardDiff: derivative, gradient
using BenchmarkTools

f(t) = sin(t)+cos(t)
df(t) = derivative(f, t)

T = 0.0

@code_typed f(T)
@code_typed df(T)

@code_native f(T)
@code_native df(T)

@btime f(T)
@btime f(T)

##
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, xs, f.(xs))
lines!(ax, xs, df.(xs))
display(fig)