using LinearAlgebra
using DifferentialEquations
using GLMakie
using Colors
using ColorSchemes

## parameters
g = 9.8
l = 1
N = 4
# m = [1, 20, 1, 1]
m = ones(N)
T = t -> zeros(N)

tspan = (0.0, 20.0)
θ₀ = zeros(N)
θ₀[N] = .1
θd₀ = zeros(N)

## dynamics
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

# function KE(x)

## solve ODE
prob = ODEProblem(f!, [θ₀; θd₀], tspan, T)
x = solve(prob)

## -------------------- plotting helper functions --------------------- ##
function thetas2xy(θ, i)
    x = -l*sum(sin.(θ[1:i]))
    y = l*sum(cos.(θ[1:i]))
    return x, y
end

function thetas2points(θvec)
    return [Point2f(theta2xy(θvec, i)) for i=1:N]
end

function colortomap(color, len)
    # Colors.lsequential_palette(color.h, )
    cmap = range(RGB(1.,1.,1.), stop=color, length=len)
    return cmap
end

function kinetic(x)
    ϕ = L * x[1:N]
    ϕd = L * x[N+1:end]
    T = 0
    for i = 1:N
        T += 1/2 * l^2 * m[i] * sum(sum([ϕd[j] * ϕd[k] * cos(ϕ[j]-ϕ[k]) for j = 1:i, k=1:i]))
    end
    return T
end

function potential(x)
    ϕ = L * x[1:N]
    V = 0
    for i = 1:N
        V += m[i] * g * l * sum([cos(ϕ[j]) for j = 1:i])
    end
    return V
end

## ---------------------------- simulate ------------------------------ #
dt = .1
tvec = tspan[1]:dt:tspan[2]
numt = length(tvec)
fps = Int(1/dt)
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

dx = zeros(2*N)
fig
record(fig, "Npend.mp4", 2:numt, framerate = fps) do frame
    t = tvec[frame]
    println(t)
    new_points = thetas2points(x(t)[1:N])
    for i = 1:N
        points[][i][] = push!(points[][i][], new_points[i])
        colors[][i][] = (numt-frame):numt
    end
    ke = kinetic(x(t))
    pe = potential(x(t))
    # f!(dx, x(t), T, t)

    θ = x(t)[1:N]; θd = x(t)[N+1:end]
    ϕ = L * θ
    ϕd = L * θd
    θdd = Linv*inv(M(ϕ)) * (-C(ϕ, ϕd) - G(ϕ) + Linv*T(t))
    println("C ", -C(ϕ, ϕd))
    println("G ", -G(ϕ))
    # println(dx)
    push!(time[], t)
    push!(KE[], ke)
    push!(PE[], pe)
    push!(totE[], ke+pe)

    sleep(1/fps)
    notify.((points[]))
    notify.([KE, PE, time])
end

