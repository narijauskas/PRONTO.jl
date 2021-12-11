# user provides: Q, R, T

function solve_riccati(A, B, Q, R, T)
    #TODO: P₁ = ?
    -A(t)'P - P*A(t) + P*B(t)*inv(R)*B(t)'P - Q
    solve(ODEProblem(riccati!, P₁, (T,0), (A,B,Q,R)))
end


function riccati!(Ṗ, P, (A,B,Q,R), t)
    Ṗ = -A(t)'P - P*A(t) + P*B(t)*inv(R)*B(t)'P - Q
end


function optKr(A, B, Q, R, T)
    # create optimal LQ regulator Kr that stabilizes around trajectory  
    P = solve_riccati(A, B, Q, R, T)# call riccati solver
    return Kᵣ = inv(R)*B'*P
end
