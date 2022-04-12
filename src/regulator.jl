# Kr = regulator((α,μ,t); Rr, Qr, fx, fu)
# Rr(t), Qr(t), fx(x,u), fu(x,u)


function riccati!(dP, P, (A,B,Q,R), t)
    K = inv(R(t))*B(t)'*P # instantenously evaluated K
    dP .= -A(t)'P - P*A(t) + K'*R(t)*K - Q(t)
end


function regulator(X, U, t, R, Q, fx, fu)
    # local, non-allocating, lazily evaluated only at values needed by ode solver
    A = t -> fx(X(t), U(t))
    B = t -> fu(X(t), U(t))
    # maybe better to use the function version below?
    # A(t) = fx(X(t), U(t))
    # B(t) = fu(X(t), U(t))

    # solve algebraic riccati eq at time T to get terminal cost
    T = last(t)
    Pt,_ = arec(A(T), B(T)inv(R(T))B(T)', Q)

    # solve differential riccati, return regulator
    P = solve(ODEProblem(riccati!, Pt, (T,0.0), (A,B,Q,R)))  
    return K(t) = inv(R(t))*B(t)'*P(t) # K as a closure with captured R,B,P
end
