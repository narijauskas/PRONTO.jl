using LinearAlgebra
using DifferentialEquations
using GLMakie

## parameters
g = 9.8
l = 1
N = 4
m = [1, 20, 1, 1]
m = [1, 1, 1, 1]
T = t -> zeros(4)

tspan = (0.0, 10.0)
dt = .05 # for animation
θ₀ = zeros(N)
θ₀[N] = .1
θd₀ = zeros(N)

## dynamics
# mass matrix M:
U = UpperTriangular(ones(N,N))
L = LowerTriangular(ones(N,N))
Linv = inv(L)

# ℳ =  (U * m) .* I(N)
ℳvec = (U * m) 
ℳ = [ℳvec[ max(i,j)] for i in 1:N, j in 1:N]
𝒞 = ϕ -> [cos(ϕ[i] - ϕ[j]) for i in 1:N, j in 1:N]
M = ϕ -> l^2 .* ℳ .* 𝒞(ϕ)

# coriolis vector C:
𝒮 = ϕ -> [sin(ϕ[i] - ϕ[j]) for i in 1:N, j in 1:N]
C = (ϕ, ϕd) -> l^2 .* ℳ .* 𝒮(ϕ) * ϕd.^2

# body force vector G:
G = ϕ -> g.*l.*ℳvec .* sin.(ϕ)

# ODE solver formulation:
function f!(dx, x, T, t)
    # L: θ -> ϕ, Linv: ϕ -> θ
    θ = x[1:N]; θd = x[N+1:end]
    ϕ = L * θ
    ϕd = L * θd
    θdd = Linv*inv(M(ϕ)) * (-C(ϕ, ϕd) - G(ϕ) + Linv*T(t))
    dx[1:N] = θd; dx[N+1:end] = θdd
end

## solve ODE
prob = ODEProblem(f!, [θ₀; θd₀], tspan, T)
x = solve(prob)

## ---------------------------- simulate ------------------------------ ##
tvec = tspan[1]:dt:tspan[2]
# time = Node(0.0)
# xplot = @lift(x($time))

points = Vector{typeof(Node(Point2f[]))}[]

colors = 

# set_theme!(theme_black())

lim = N*l
fig, ax, l = lines(points[1], color = colors,
    colormap = :inferno, transparency = true,
    axis = (; limits = (-lim, lim, -lim, lim)))

for i = 2:N
    lines!

function phi2xy(ϕ, i)
    x = -l*sum(sin.(ϕ[1:i]))
    y = l*sum(cos.(ϕ[1:i]))
    return x, y
end

function phis2points(ϕvec)
    return [Point2f(phi2xy(ϕvec, i)) for i=1:N]
end

record(fig, "Npend.mp4", tvec, framerate = 30) do t
    new_points = phis2points(x(t)[1:N])
    push!(points[], step!(attractor))
    push!(colors[], frame)
    ax.azimuth[] = 1.7pi + 0.3 * sin(2pi * frame / 120)
    notify.((points, colors))
    l.colorrange = (0, frame)
end