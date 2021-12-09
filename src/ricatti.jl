
function solve_ricatti(A, B, Q, R)
    #TODO: P₁ = ?
    solve(ODEProblem(ricatti!, P₁, T:Δt:0, (A,B,Q,R)))
end


function ricatti!(Ṗ, P, params, t)
    (A,B,Q,R) = params
    Ṗ = -A(t)'P - P*A(t) + P*B(t)*inv(R)*B(t)'P - Q
end


function optKr(A, B, Q, R)
    # create optimal LQ regulator Kr that stabilizes around trajectory  
    P = solve_ricatti(A, B, Q, R)# call riccati solver
    return Kᵣ = inv(R)*B'*P
end
