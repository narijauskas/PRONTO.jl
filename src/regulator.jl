# Kr = regulator((α,μ,t); Rr, Qr, fx, fu)
# Rr(t), Qr(t), fx(x,u), fu(x,u)


function riccati!(dP, P, (A,B,Q,R), t)
    K = inv(R(t))*B(t)'*P # instantenously evaluated K
    dP .= -A(t)'P - P*A(t) + K'*R(t)*K - Q(t)
end

# function regulator(X, U, t, R, Q, fx, fu)

function regulator(x, u, model)
    # local, non-allocating, lazily evaluated only at values needed by ode solver
    # A = t->model.fx(x(t), u(t))
    # B = t->model.fu(x(t), u(t))
    # timeseries provide type stability
    A = Timeseries(t->model.fx(x(t), u(t)))
    B = Timeseries(t->model.fu(x(t), u(t)))

    # solve algebraic riccati eq at time T to get terminal cost
    T = last(model.t); R = model.Rr; Q = model.Qr
    Pt,_ = arec(A(T), B(T)inv(R(T))B(T)', Q(T))


    # solve differential riccati, return regulator
    P = Timeseries(solve(ODEProblem(riccati!, Pt, (T,0.0), (A,B,Q,R))))
    Kr = Timeseries(t->inv(R(t))*B(t)'*P(t))
    # Kr = t->inv(R(t))*B(t)'*P(t) # K as a closure with captured R,B,P
    return Kr
end


# regulator goal
# map ξ or φ -> Kr


# regulator setup

function riccati!(dP, P, model, t)
    Rr = model.Rr(t)
    Qr = model.Qr(t)
    A = model.fx(ξ.x(t),ξ.u(t))
    B = model.fu(ξ.x(t),ξ.u(t))

    K = inv(Rr)*B'*P # instantenously evaluated K
    dP .= -A'P - P*A + K'*Rr*K - Qr
end

function PT(model, ξ)
    Rr = model.Rr(T)
    Qr = model.Qr(T)
    A = model.fx(ξ.x(T),ξ.u(T))
    B = model.fu(ξ.x(T),ξ.u(T))

    PT,_ = arec(A, B*inv(Rr)*B', Qr)
    return PT
end

# PT,_ = arec(A(T), B(T)inv(Rr(T))B(T)', Qr(T))
Pr = Interpolant(?, t)
Pr_integrator = init(ODEProblem(riccati!, PT(model,ξ), (T,0.0)), Tsit5())
# Kr(t) = inv(Rr(t))*B(t)'*P(t)
function Kr(model,ξ,Pr,t)
    Rr = model.Rr(t)
    B = model.fu(ξ.x(t),ξ.u(t))
    return inv(Rr)*B'*Pr(t) # instantenously evaluated K
    # return inv(Rr(t))*B(t)'*P(t)
end




# regulator loop
# A = model.fx(ξ.x(t),ξ.u(t))
# B = model.fu(ξ.x(t),ξ.u(t))
# PT,_ = arec(A(T), B(T)inv(Rr(T))B(T)', Qr(T))

resolve!(Pr, Pr_integrator, PT(model,ξ))
# now Kr is up to date

