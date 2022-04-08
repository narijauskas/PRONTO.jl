# sketch of high-level PRONTO api

# create t vector, pre-allocate x(t) & u(t) (defines output dimensions)
# each iteration modifies the trajectory in-place
pronto!(x, u, t, fxns...; opts...)
pronto!(ξ, fxns...; opts...)
pronto!(ξ, f, l, p; opts...)

# trajectory type
ξ.x
ξ.u
ξ.t
# maybe: built in projection?


# maybe: debug/diagnostic version -> sa ves each iteration step
# pronto(x, u, t, fxns...; opts...)

# user provided functions:
f(x,u,t)
l(x,u,t)
p(x)


# helper/convenience functions:
project()
# ways to make f,l,p
ξ = ξ_init(x_eq, u_guess, t)


