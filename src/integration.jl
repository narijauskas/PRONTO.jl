


using DifferentialEquations
ts = 0.0:0.01:10.0
p = Interpolant(t->1.01, ts)
f1(u,p,t) = p(t)*u
u0 = 1/2
tspan = extrema(ts)
prob = ODEProblem(f1,u0,tspan, p)

integrator = init(prob, Tsit5())


X = Interpolant(t->0.0, ts)
Xfxn(t) = sin(X(t))


resolve!(X,integrator,u0)
X
update!(t->1.03, p)
resolve!(X,integrator,u0)
Xfxn(10)

function resolve!(X,integrator,x0)
    reinit!(integrator,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(integrator, X.t))
        X[i] = x
        # @show x,t
        #maybe: check X.t[i] == t?
    end
    return nothing
end




