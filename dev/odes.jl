# can we put a time-variant parameter into an ODEProblem?

using DataInterpolations, BenchmarkTools
using DifferentialEquations


t = 0:1:10
x = sin.(t)
xq = LinearInterpolation(x,t)

τ = 0:0.1:10
lines(t,x)
lines!(τ, xq.(τ))


# does it work with ODEs?
du(u,p,t) = p(t)
sol = solve(ODEProblem(du, 0, extrema(t), xq))

lines(t, sol(t).u)
lines!(t, xq.(t))


#TODO:
#CITE:
# does this work with a matrix?
@btime y = collect([sin(t) cos(t)] for t in t) # 0.5 μs
@btime LinearInterpolation(y,t) # 0.7 μs
Y = LinearInterpolation(y,t) # 0.7 μs
@btime Y.(τ) # 14 μs
yq = similar(Y.(τ))
@btime map!(t->Y(t), yq, τ) # 17 μs

# does the matrix work with ODEs?
sol = solve(ODEProblem(du, [0 0], extrema(t), Y))

lines(t, map(t->sol(t)[1],t))
lines!(t, map(t->sol(t)[2],t))

# lines!(t, xq.(t))
lines!(t, map(t->Y(t)[1],t))
lines!(t, map(t->Y(t)[2],t))


# can I change p?
z = collect([cos(t)*sin(t) 0] for t in t)
Z = LinearInterpolation(z,t)
Y.u .= Z.u

du(u,p,t) = p(t)
prob = ODEProblem(du, [0 0], extrema(t), Y) # 26.8 μs
sol = solve(prob) # 108 μs


lines(t, map(t->sol(t)[1],t))
lines!(t, map(t->sol(t)[2],t))

lines!(t, map(t->Y(t)[1],t))
lines!(t, map(t->Y(t)[2],t))
lines!(t, map(t->Z(t)[1],t))
lines!(t, map(t->Z(t)[2],t))



