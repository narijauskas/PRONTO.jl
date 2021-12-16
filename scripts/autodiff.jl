using ForwardDiff
using ForwardDiff: derivative, gradient
using BenchmarkTools

ff(t) = cos(t) + cos(t)
df(t) = derivative(ff, t)

T = 0.0

@code_typed ff(T)
@code_typed df(T)

@code_native ff(T)
@code_native df(T)

@btime ff(T)
@btime df(T)

##
fig = Figure(); ax = Axis(fig[1,1])
lines!(ax, xs, f.(xs))
lines!(ax, xs, df.(xs))
display(fig)