# user must define:

# a model type, with any scalar parameters as fields
struct Model <: Pronto.Model
end

f(θ,x,u,t)
l(θ,x,u,t)
m(θ,x,t)

Rr(θ,x,u,t)
Qr(θ,x,u,t)
# Pr(θ,x) ... for final condition?

# teach PRONTO the model and build a model kernel
@configure Model

# create an instance of the model with parameters
θ = Model(params...)

x0
guess(...)->αg,μg
# T = [t0:dt:tf]
T = (t0,tf) # if variable timestep