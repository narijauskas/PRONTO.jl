# ζ = (z,v) = search_direction(φ..., Kr, model)

# backward integration
function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end

function costate_dynamics!(dx, x, (A,B,a,b,K), t)
    dx .= -(A(t)-B(t)*K(t))'*x - a(t) + K(t)'*b(t)
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

    

# 1. create integrator (with model? at package level? at model construction?)

# 2. re-init integrator at [x0]
# 3. for each t in ξ.t
    # 3.1 solve integrator to t 
    # 3.2 update value of interpolant (which is NOT used to solve #3)

# 4. use value for next solution

#MAYBE: create solver object, containing the "internal" solutions
# eg, trajectories like ξ/ζ, and internal solutions like Kr,P,r,q

function search_direction(x,u,Kr,model)
    #MAYBE: split this into parts?

    # --------------- define tangent space --------------- #
    #MAYBE: condense this into a separate function?

    A = Timeseries(t->model.fx(x(t), u(t)))
    B = Timeseries(t->model.fu(x(t), u(t)))
    a = Timeseries(t->model.lx(x(t), u(t)))
    b = Timeseries(t->model.lu(x(t), u(t)))
    Q = Timeseries(t->model.lxx(x(t), u(t)))
    R = Timeseries(t->model.luu(x(t), u(t)))
    S = Timeseries(t->model.lxu(x(t), u(t)))


    # --------------- backward integration --------------- #

    T = last(model.t)
    PT = model.pxx(model.x_eq) # use x_eq
    rT = model.px(model.x_eq)

    P = Timeseries(solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S))))
    Ko = Timeseries(t->R(t)\(S(t)'+B(t)'*P(t)))

    #YO: fails with Rosenbrock23()
    r = Timeseries(solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko))))
    vo = Timeseries(t->(-R(t)\(B(t)'*r(t)+b(t))))
    
    
    qT = rT # use x_eq
    q = Timeseries(solve(ODEProblem(costate_dynamics!, qT, (T,0.0), (A,B,a,b,Kr))))
    q = Timeseries(t->q(t))



    # --------------- forward integration --------------- #

    z0 = 0 .* model.x_eq
    z = Timeseries(solve(ODEProblem(update_dynamics!, z0, (0.0,T), (A,B,Ko,vo))))
    # z = Timeseries(t->z(t))
    v = Timeseries(t->(-Ko(t)*z(t)+vo(t)))
    ζ = (z,v)

    # --------------- step type --------------- #
    #TODO: newton step vs gradient descent
    # lagrangians?

    #YO: temporary
    Ro = R
    Qo = Q
    So = S

    # --------------- cost derivatives --------------- #
    y0 = [0;0]
    y = Timeseries(solve(ODEProblem(cost_derivatives!, y0, (0.0,T), (z,v,a,b,Qo,So,Ro))))
    Dh = y(T)[1] + rT'*z(T)
    D2g = y(T)[2] + z(T)'*PT*z(T)
    # Dh  = update.y1.Data(end)+rT'*update.z.Data(end,:)';
    # Dg2 = update.y2.Data(end)+update.z.Data(end,:)*PT*update.z.Data(end,:)';

    return ζ,Dh
end


# R₀ = R
# Q₀ = Q
# S₀ = S

# R₀ = t -> R(t) .+ mapreduce((qk,fk)->qk*fk, sum, q(t), fuu(X1(t), U1(t)))

# R₀ = t -> R(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fuu(X1(t), U1(t))))
# Q₀ = t -> Q(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxx(X1(t), U1(t))))
# S₀ = t -> S(t) .+ sum(map((qk,fk) -> qk*fk, q(t), Main.fxu(X1(t), U1(t))))
