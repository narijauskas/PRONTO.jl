export regulator

struct Regulator{M,T1,T2,T3}
    θ::M
    α::T1
    μ::T2
    Pr::T3
end


(Kr::Regulator)(t) = Kr(Kr.α(t), Kr.μ(t), t)
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




regulator(θ,φ,τ) = regulator(θ, φ.x, φ.u, τ)
# design the regulator, solving dPr_dt
function regulator(θ::Model{NX,NU}, α, μ, (t0,tf)) where {NX,NU}
    #FUTURE: Pf provided by user or auto-generated as P(α,μ,θ)
    Pf = SMatrix{NX,NX,Float64}(I(NX))
    Pr = ODE(dPr_dt, Pf, (tf,t0), (θ,α,μ))
    Regulator(θ,α,μ,Pr)
end



function dPr_dt(Pr,(θ,α,μ),t)#(M, out, θ, t, φ, Pr)
    α = α(t)
    μ = μ(t)

    Ar = fx(α,μ,t,θ)
    Br = fu(α,μ,t,θ)
    Qr = Q(α,μ,t,θ)
    Rr = R(α,μ,t,θ)
    
    - Ar'Pr - Pr*Ar + Pr'Br*(Rr\Br'Pr) - Qr
end


