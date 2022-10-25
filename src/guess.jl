
massmatrix(M) = cat(diagm(ones(nx(M))), zeros(nu(M)); dims=(1,2))

function guess_zi(M,θ,x0,u0,t0,tf)
    odefn = ODEFunction(zi_dyn!; mass_matrix=massmatrix(M))
    prob = ODEProblem(odefn, [x0;u0], (t0,tf), (M,θ))
    Trajectory(prob, nx(M), nu(M))
end

function zi_dyn!(dξ,ξ,(M,θ),t)
    dx = @view dξ[1:nx(M)]
    du = @view dξ[(nx(M)+1):end]
    x = @view ξ[1:nx(M)]
    u = @view ξ[(nx(M)+1):end]

    f!(M,dx,θ,t,x,u)
    du .= .- u
end