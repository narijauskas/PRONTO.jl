# reinitialize and solve integrator, saving steps to X
function resolve!(X,integrator,x0)
    reinit!(integrator,x0)
    for (i,(x,t)) in enumerate(TimeChoiceIterator(integrator, X.t))
        X[i] = x
        # @show x,t
        #maybe: check X.t[i] == t?
    end
    return nothing
end


# setup
using DifferentialEquations
using DifferentialEquations: init # otherwise the linter complains
ts = 0.0:0.01:10.0
x0 = zeros(3)

@variables x[1:3]
q = x->cos.(collect(x))
qx = jacobian(x,q,x)
model = MStruct()
model.qx = qx

function qx_manual(x)
    return [
        -sin(x[1]) 0 0;
        0 -sin(x[2]) 0;
        0 0 -sin(x[3]);
    ]
end
X = Interpolant(t->qx_manual(x0), ts)
Xfxn(t) = sin.(X(t)) # a function that captures X

# p = Interpolant(t->1.01, ts)
# p = cos
fxn = (x,model,t)->model.qx(x)
X_intg = init(ODEProblem(fxn,model.qx(x0),extrema(ts),model), Tsit5())

@time resolve!(X,X_intg,qx(x0))
@benchmark Xfxn(3)
@benchmark resolve!(X,X_intg,qx(x0))

# quick benchmark result: symbolically derived jacobian stored in an MStruct
# performs as well as a directly-defined function


