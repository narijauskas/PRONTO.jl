# Kr = regulator((α,μ,t); Rr, Qr, fx, fu)
# Rr(t), Qr(t), fx(x,u), fu(x,u)


function riccati!(dP, P, (A,B,Q,R), t)
    K = inv(R(t))*B(t)'*P # instantenously evaluated K
    dP .= -A(t)'P - P*A(t) + K'*R(t)*K - Q(t)
end

# function regulator(X, U, t, R, Q, fx, fu)

function regulator(X, U, model)
    # local, non-allocating, lazily evaluated only at values needed by ode solver
    # A = t -> model.fx(X(t), U(t))
    # B = t -> model.fu(X(t), U(t))
    # timeseries provide type stability
    A = Timeseries(t -> model.fx(X(t), U(t)), model.t)
    B = Timeseries(t -> model.fu(X(t), U(t)), model.t)


    # solve algebraic riccati eq at time T to get terminal cost
    T = last(model.t); R = model.R; Q = model.Q
    Pt,_ = arec(A(T), B(T)inv(R(T))B(T)', Q(T))


    # solve differential riccati, return regulator
    P = solve(ODEProblem(riccati!, Pt, (T,0.0), (A,B,Q,R)), Rosenbrock23(), dt=0.001)
    Kr = Timeseries(t->inv(R(t))*B(t)'*P(t), model.t)
    # P = LinearInterpolation(hcat(P.(t)...),t) 
    # Kr = t->inv(R(t))*B(t)'*P(t) # K as a closure with captured R,B,P
    return Kr
end
