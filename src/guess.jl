
#MAYBE: convert to ξ0?
function guess_zi(M::Model{NX,NU,NΘ},θ,x0,u0,t0,tf) where {NX,NU,NΘ}
    ODE(zi_dyn!, [x0;u0], (t0,tf), (M,θ), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
end
function zi_dyn!(dξ,ξ,(M,θ),t)
    dx = @view dξ[1:nx(M)]
    du = @view dξ[(nx(M)+1):end]
    # x = @view ξ[1:nx(M)]
    u = @view ξ[(nx(M)+1):end]

    f!(M,dx,θ,t,ξ)
    du .= .- u
end


export smooth, guess_φ

smooth(t, x0, xf, T) = @. (xf - x0)*(tanh((2π/T)*t - π) + 1)/2 + x0

# φg can be a closure, eg.
# φg = @closure t->[smooth(t,x0,xf,T); 0.0]
function guess_φ(M::Model{NX,NU,NΘ},θ,ξ0,t0,tf,φg) where {NX,NU,NΘ}
    Pr_f = diagm(ones(NX))
    Pr = ODE(Pr_ode, Pr_f, (tf,t0), (M,θ,φg), ODEBuffer{Tuple{NX,NX}}())

    φ = ODE(ξ_ode, ξ0, (t0,tf), (M,θ,φg,Pr), ODEBuffer{Tuple{NX+NU}}(); dae=dae(M))
    return φ
end

