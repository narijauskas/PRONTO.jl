# helpful extras, eg.

# symbolic representation of PRONTO functions
symbolic(PRONTO.f, θ)

# sweep parameters -> create 10 θ instances with varying k and @spawn threads to solve each
sweep(θ; k=[1:10]) 

# sample a functionwrapper -> return vector of SVectors
sample(x, 0:0.001:10)

# simulate various dynamics, eg. OL or under an input (u) or control law (K,α,μ)
simulate(θ, (t0,tf))
