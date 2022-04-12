
# trajectory object

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