# export regulator

struct Regulator{M,Φ,P}
    θ::M
    φ::Φ
    Pr::P
end

(Kr::Regulator)(t) = Kr(Kr.φ.x(t), Kr.φ.u(t), t)
(Kr::Regulator)(α, μ, t) = Kr(α, μ, t, Kr.θ)

function (Kr::Regulator)(α, μ, t, θ)
    Pr = Kr.Pr(t)
    Rr = R(α,μ,t,θ)
    Br = fu(α,μ,t,θ)

    Rr\Br'Pr
end


nx(Kr::Regulator) = nx(Kr.θ)
nu(Kr::Regulator) = nu(Kr.θ)
extrema(Kr::Regulator) = extrema(Kr.Pr)
eachindex(Kr::Regulator) = OneTo(nu(Kr)*nx(Kr))
show(io::IO, Kr::Regulator) = println(io, preview(Kr))




# regulator(θ,φ,τ) = regulator(θ, φ.x, φ.u, τ)
# design the regulator, solving dPr_dt
 function regulator(θ::Model{NX,NU}, φ, τ) where {NX,NU}
    t0,tf = τ
    #FUTURE: Pf provided by user or auto-generated as P(α,μ,θ)
    # α 
    # Pf = SMatrix{NX,NX,Float64}(I(NX) - φ.x(tf)*(φ.x(tf))')
    Pf = SMatrix{NX,NX,Float64}(I(NX))
    Pr = ODE(dPr_dt, Pf, (tf,t0), (θ,φ))
    Regulator(θ,φ,Pr)
end



function dPr_dt(Pr,(θ,φ),t)#(M, out, θ, t, φ, Pr)
    α = φ.x(t)
    μ = φ.u(t)

    Ar = fx(α,μ,t,θ)
    Br = fu(α,μ,t,θ)
    Qr = Q(α,μ,t,θ)
    Rr = R(α,μ,t,θ)
    
    return - Ar'Pr - Pr*Ar + Pr'Br*(Rr\Br'Pr) - Qr
end


