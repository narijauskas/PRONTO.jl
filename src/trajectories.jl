
# trajectory object
# technically a timeseries

# for now, forces sampling to interpolation?
# or, keeps it generic to any callable thing

# φₜ and t

#note: this is a sloppy initial implementation that should be optimized later


# 



Trajectory(v,t) = Trajectory(length(v(first(t))), v, t)

# general
Base.length(φ::Trajectory) = φ.n
Base.show(io::IO, φ::Trajectory) = println(io, "Trajectory with $(length(φ)) states")

# φ(t) = v(t)
(φ::Trajectory)(t) = φ.v(t)



# resample!(φ, t)

# scalar multiplication



# not just trajectory, but timeseries

# plotable, callable, have dimensionality
# helps wrap/abstract away some of the madness that are the current return types
