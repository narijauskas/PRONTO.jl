# see jupyter notebook
using LinearAlgebra
using DifferentialEquations
using GLMakie
using Colors
using ColorSchemes


## -------------------- build and solve ODE --------------------- ##

# parameters
g = 9.8
l = 1
N = 2
# m = [1, 20, 1, 1]
m = ones(N)
T = t -> zeros(N)

tspan = (0.0, 40.0)
θ₀ = zeros(N)
θ₀[N] = π/2
θd₀ = zeros(N)

# dynamics
# mass matrix M:
U = UpperTriangular(ones(N,N))
L = LowerTriangular(ones(N,N))
Linv = inv(L)
v1 = ones(N)

# ℳ =  (U * m) .* I(N)
ℳvec = (U * m) 
ℳ = [ℳvec[ max(i,j)] for i in 1:N, j in 1:N]
𝒞 = ϕ -> [cos(ϕ[i] - ϕ[j]) for i in 1:N, j in 1:N]
M = ϕ -> l^2 .* ℳ .* 𝒞(ϕ)

# coriolis vector C:
𝒮 = ϕ -> [sin(ϕ[i] - ϕ[j]) for i in 1:N, j in 1:N]
C = (ϕ, ϕd) -> l^2 .* ℳ .* 𝒮(ϕ) .* (v1*ϕd' - 2ϕd*v1') * ϕd

# body force vector G:
G = ϕ -> @. -g*l*ℳvec*sin(ϕ)
# G = ϕ -> g.*l.*ℳvec .* sin.(ϕ)


# ODE solver formulation:
function f!(dx, x, T, t)
    # L: θ -> ϕ, Linv: ϕ -> θ
    θ = x[1:N]; θd = x[N+1:end]
    ϕ = L * θ
    ϕd = L * θd
    θdd = Linv*inv(M(ϕ)) * (-C(ϕ, ϕd) - G(ϕ) + Linv*T(t))
    dx[1:N] = θd; dx[N+1:end] = θdd
end

function fϕ!(dx, x, T, t)
    # x = [ϕ,dϕ], dx = [dϕ,ddϕ]
    ϕ = x[1:N]
    dϕ = dx[1:N] .= x[N+1:end]
    dx[N+1:end] .= inv(M(ϕ)) * (-C(ϕ, dϕ) - G(ϕ) + Linv*T(t))
end

# function KE(x)

# solve ODE
# prob = ODEProblem(f!, [θ₀; θd₀], tspan, T)
# x = solve(prob)


# prob = ODEProblem(fϕ!, [θ₀; θd₀], tspan, T)
# x = solve(prob)

## -------------------- plotting helper functions --------------------- ##
function phi2xy(ϕ, i)
    x = -l*sum(sin.(ϕ[1:i]))
    y = l*sum(cos.(ϕ[1:i]))
    return x, y
end

function phis2points(ϕvec)
    return [Point2f(phi2xy(ϕvec, i)) for i=1:N]
end

function colortomap(color, len)
    # Colors.lsequential_palette(color.h, )
    cmap = range(RGB(1.,1.,1.), stop=color, length=len)
    return cmap
end

function kinetic(x)
    ϕ = x[1:N]
    ϕd = x[N+1:end]
    T = 0
    for i = 1:N
        T += 1/2 * l^2 * m[i] * sum(sum([ϕd[j] * ϕd[k] * cos(ϕ[j]-ϕ[k]) for j = 1:i, k=1:i]))
    end
    return T
end

function potential(x)
    ϕ = x[1:N]
    V = 0
    for i = 1:N
        V += m[i] * g * l * sum([cos(ϕ[j]) for j = 1:i])
    end
    return V
end


function ϕdϕ(x)
    n = Int(length(x)/2)
    return (x[1:n], x[n+1:end])
end

function getϕ(x)
    n = Int(length(x)/2)
    return x[1:n]
end

## ---------------------------- simulate ------------------------------ #
fps = 30
dt = 1/fps
tvec = x.t[1]:dt:x.t[end]
numt = length(tvec)
ICpoints = phis2points(θ₀)
points = Node( [Node( [Point2f(ICpoints[i])] ) for i=1:N] )

time = Node( [Float64(0.0)] )
KE = Node( [Float64(kinetic([θ₀; θd₀]))] )
PE = Node( [Float64(potential([θ₀; θd₀]))] )
totE = Node( [Float64(KE[][1]+PE[][1])])

colorsc = ColorSchemes.hawaii
colorvec = [colorsc[i/N] for i = 0:N-1]
colormaps = [colortomap(colorvec[i], length(tvec)) for i in 1:N]
colors = Node( [Node( [Int(0)] ) for i=1:N] )

lim = N*l
fig = Figure()
ax1 = Axis(fig[1, 1], limits=(-lim, lim, -lim, lim))
ax2 = Axis(fig[1, 2], limits = (tspan[1], tspan[2], nothing, nothing))
scatter!(ax1, points[][1], color = colors[][1], colormap = colormaps[1],
    transparency = true)

for i = 2:N
    scatter!(ax1, points[][i], color = colors[][i], colormap = colormaps[i])
end

lines!(ax2, time, KE, color = :red)
lines!(ax2, time, PE, color = :blue)
lines!(ax2, time, totE, color = :purple)

fig
##

record(fig, "Npend.mp4", 2:numt, framerate = fps) do frame
    t = tvec[frame]
    println(t)
    new_points = phis2points(x(t)[1:N])
    for i = 1:N
        points[][i][] = push!(points[][i][], new_points[i])
        colors[][i][] = (numt-frame):numt
    end
    ke = kinetic(x(t))
    pe = potential(x(t))
    # f!(dx, x(t), T, t)

    ϕ = x(t)[1:N]; ϕd = x(t)[N+1:end]
    θdd = inv(M(ϕ)) * (-C(ϕ, ϕd) - G(ϕ) + Linv*T(t))
    println("KE ", ke)
    println("PE ", pe)
    # println(dx)
    push!(time[], t)
    push!(KE[], ke)
    push!(PE[], pe)
    push!(totE[], ke+pe)

    sleep(1/fps)
    notify.((points[]))
    notify.([KE, PE, time])
end

## ---------------------------- useful plots ------------------------------ ##

x1 = solve(ODEProblem(fϕ!, [θ₀; θd₀], tspan, T), Rosenbrock23()) # matlab ode23s
x2 = solve(ODEProblem(fϕ!, [θ₀; θd₀], tspan, T), TRBDF2())
x3 = solve(ODEProblem(fϕ!, [θ₀; θd₀], tspan, T), BS3()) # matlab ode23
x = x1
x = x2
x = x3
##
t = x.t[1]:0.01:x.t[end]
ϕt = [map(tx->x(tx)[ix], t) for ix in 1:N]
dϕt = [map(tx->x(tx)[ix], t) for ix in N+1:2N]


fig = Figure(); display(fig)

ax = Axis(fig[1,1]; title="ϕ(t)")
for ϕ in ϕt
    lines!(ax, t, ϕ)
end

ax = Axis(fig[2,1]; title="dϕ(t)")
for dϕ in dϕt
    lines!(ax, t, dϕ)
end

ax = Axis(fig[1,2]; title="G(t)")
for ix in 1:N
    lines!(ax, t, map(tx->G(x(tx)[1:N])[ix], t))
end


# map(tx->C(ϕdϕ(x(tx))...), t)
ax = Axis(fig[2,2]; title="C(t)")
for ix in 1:N
    lines!(ax, t, map(tx->C(ϕdϕ(x(tx))...)[ix], t))
end

# @. potential(getϕ(x(t)))
ax = Axis(fig[1:2,3]; title="energy")
lines!(ax, t, @. potential(x(t)))
lines!(ax, t, @. kinetic(x(t)))
lines!(ax, t, @. potential(x(t)) + kinetic(x(t)))