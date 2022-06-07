


using DifferentialEquations
f1(u,p,t) = 1.01*u
u0 = 1/2
ts = 0.0:0.01:1.0
tspan = extrema(ts)
prob = ODEProblem(f1,u0,tspan)

integrator = init(prob, Tsit5())


X = Interpolant(t->0.0, ts)


function resolve!(X,integrator,x0)
    reinit!(integrator,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(integrator, X.t))
        X[i] = x
        @show x,t
        #maybe: check X.t[i] == t?
    end
    return nothing
end




