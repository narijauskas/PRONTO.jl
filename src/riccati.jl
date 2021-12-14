
function riccati!(Ṗ, P, (A,B,Q,R), t)
    Ṗ = -A(t)'P - P*A(t) + P*B(t)*inv(R)*B(t)'P - Q
end


# create an optimal LQR controller around the linearization of f(ξ)
function optKr(f, ξ, Q, R, T)
    A = Jx(f, ξ)
    B = Ju(f, ξ)
    P₁,_ = arec(A(T), B(T)inv(R)B(T)', Q) # solve algebraic riccati eq at time T
    solve(ODEProblem(riccati!, P₁, (T,0), (A,B,Q,R))) # solve differential riccati
    return Kᵣ = inv(R)*B'*P
end
