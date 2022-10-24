# helpful extras, eg.
using Symbolics

# symbolic representation of PRONTO functions
function symbolic(f::Function, M::Model)
    @variables x[1:nx(M)] 
    @variables u[1:nu(M)] 
    @variables t
    @variables θ[1:nθ(M)]
    f(M,x,u,t,θ)
end

function symbolic(f::typeof(dPr), M::Model)
    @variables α[1:nx(M)] 
    @variables μ[1:nu(M)] 
    @variables t
    @variables θ[1:nθ(M)]
    @variables Pr[1:nx(M),1:nx(M)]
    f(M,α,μ,t,θ,Pr)
end
# sweep parameters -> create 10 θ instances with varying k and @spawn threads to solve each
# sweep(θ; k=[1:10]) 

# sample a functionwrapper -> return vector of SVectors
# sample(x, 0:0.001:10)

# simulate various dynamics, eg. OL or under an input (u) or control law (K,α,μ)
# simulate(θ, (t0,tf))
