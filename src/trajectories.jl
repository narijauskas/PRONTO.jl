
# trajectory object
# technically a timeseries

# for now, forces sampling to interpolation?
# or, keeps it generic to any callable thing

# φₜ and t

#note: this is a sloppy initial implementation that should be optimized later

struct Trajectory
    n::Int # number of states
    v # any call-able thing which returns an array length n
    t::AbstractVector # time range of validity
end

Trajectory(v,t) = Trajectory(length(v(first(t))), v, t)

# general
Base.length(φ::Trajectory) = φ.n
Base.show(io::IO, φ::Trajectory) = println(io, "Trajectory with $(length(φ)) states")

# φ(t) = v(t)
(φ::Trajectory)(t) = φ.v(t)

# φ[1](t) # fundamentally inefficient with the current implementation
Base.getindex(φ::Trajectory, i::Int) = 1 <= i <= length(φ) ? (return t->φ(t)[i]) : throw(BoundsError(φ, i))

# for v in φ; v(t); end
Base.iterate(φ::Trajectory, state=1) = state <= length(φ) ? (return (φ[state], state+1)) : (return nothing)


# resample!(φ, t)


t = ξ.t
fig = Figure(); ax = Axis(fig[1,1])
for v in ξ
    lines!(ax, t, v.(t))
end


f(x(t),u(t),t)

Trajectory(τ->f(τ), t)


# scalar multiplication




# constructor from (x,u,t)
# constructor from ODESolution
# constructor from f,t?


# not just trajectory, but timeseries

# plotable, callable, have dimensionality
# helps wrap/abstract away some of the madness that are the current return types


Kr = regulator(...)
plot(Kr) # will break 
plot(Timeseries(Kr, t))
plot(Timeseries(Kr.(t)))