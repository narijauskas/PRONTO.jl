

function guess_zi(M,θ,x0,u0,t0,tf)
    Trajectory(M, zi_dyn!,[x0;u0], (t0,tf), (M,θ))
end
function zi_dyn!(dξ,ξ,(M,θ),t)
    dx = @view dξ[1:nx(M)]
    du = @view dξ[(nx(M)+1):end]
    # x = @view ξ[1:nx(M)]
    u = @view ξ[(nx(M)+1):end]

    f!(M,dx,θ,t,ξ)
    du .= .- u
end