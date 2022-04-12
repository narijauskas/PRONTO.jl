
# trajectory object

# constructor from (x,u,t)
# constructor from ODESolution
# constructor from f,t?


# not just trajectory, but timeseries



Kr = regulator(...)
plot(Kr) # will break
plot(Timeseries(Kr, t))
plot(Timeseries(Kr.(t)))