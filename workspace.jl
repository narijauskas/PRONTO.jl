# a translation/analogue of drive_ipend.m

# set up a function space:
dt = 0.01; t0 = 0.0; t1 = 10.0
T0 = t0:dt:t1

Φ₀ = @. tanh(T0-t1/2)*(1-tanh(T0-t1/2)^2)
maximum(Φ₀)
