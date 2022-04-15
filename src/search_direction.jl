# ζ = (z,v) = search_direction(...)


# tangent space
# A = t->Main.fx(X1(t), U1(t))
# B = t->Main.fu(X1(t), U1(t))
# a = t->Main.l_x(X1(t), U1(t))
# b = t->Main.l_u(X1(t), U1(t))
# Q = t->Main.lxx(X1(t), U1(t))
# R = t->Main.luu(X1(t), U1(t))
# S = t->Main.lxu(X1(t), U1(t))


## --------------------------- backward integration --------------------------- ##

# backward integration
function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + collect(K(t)'*b(t))
end


# forward integration
function update_dynamics!(dz, z, (A,B,Ko,vo), t)
    v = -Ko(t)*z+vo(t)
    dz .= A(t)*z+B(t)*v
end

function cost_derivatives!(dy, y, (z,v,a,b,Qo,So,Ro), t)
    dy[1] = a(t)'*z(t)+b(t)'*v(t)
    dy[2] = z(t)'*Qo(t)*z(t)+2*z(t)'*So(t)*v(t)+v(t)'*Ro(t)*v(t)
end

    

function search_direction(x,u,t,model,Kr,x_eq)
    # define tangent space
    A = t->Main.fx(x(t), u(t))
    B = t->Main.fu(x(t), u(t))
    a = t->Main.l_x(x(t), u(t))
    b = t->Main.l_u(x(t), u(t))
    Q = t->Main.lxx(x(t), u(t))
    R = t->Main.luu(x(t), u(t))
    S = t->Main.lxu(x(t), u(t))


    T = last(t)
    PT = model[:pxx](x_eq) # use x_eq

    P = solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S)), dt=1e-3)
    # Ko = Trajectory(τ->inv(R(τ))*(S(τ)'+B(τ)'*P(τ)), t)
    Ko(t) = R(t)\(S(t)'+B(t)'*P(t))

    rT = model[:px](x_eq) # use x_eq
    r = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)), Rosenbrock23(), dt=0.001)
    # vo = tau(τ->(-R(τ)\(B(τ)'*r(τ)+b(τ))), t)
    vo(t) = -R(t)\(B(t)'*r(t)+b(t))
    # vo = Trajectory(τ->(-R(τ)\(B(τ)'*r(τ)+b(τ))), t)

    qT = rT # use x_eq
    q = solve(ODEProblem(costate_dynamics!, qT, (T,0.0), (A,B,a,b,Kr)), dt=0.001)
    # q = Trajectory(τ->q(τ), t)

    z0 = 0 .* x_eq
    z = solve(ODEProblem(update_dynamics!, z0, (0.0,T), (A,B,Ko,vo)))
    v = t -> -Ko(t)*z(t)+vo(t)

    #YO: temporary
    Ro = R
    Qo = Q
    So = S

    y0 = [0;0]
    y = solve(ODEProblem(cost_derivatives!, y0, (0.0,T), (z,v,a,b,Qo,So,Ro)))
    Dh = y(T)[1] + rT'*z(T)
    D2g = y(T)[2] + z(T)'*PT*z(T)
    # Dh  = update.y1.Data(end)+rT'*update.z.Data(end,:)';
    # Dg2 = update.y2.Data(end)+update.z.Data(end,:)*PT*update.z.Data(end,:)';

    return Ko,vo,q,z,v,y,Dh,D2g
end


# R₀ = R
# Q₀ = Q
# S₀ = S

# R₀ = t -> R(t) .+ mapreduce((qk,fk)->qk*fk, sum, q(t), fuu(X1(t), U1(t)))

# R₀ = t -> R(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fuu(X1(t), U1(t))))
# Q₀ = t -> Q(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxx(X1(t), U1(t))))
# S₀ = t -> S(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxu(X1(t), U1(t))))

## --------------------------- forward integration --------------------------- ##


