# export regulator

struct Regulator{M,Φ,P}
    θ::M
    φ::Φ
    Pr::P
end

(Kr::Regulator)(t) = Kr(Kr.φ.x(t), Kr.φ.u(t), t)
(Kr::Regulator)(α, μ, t) = Kr(Kr.θ, α, μ, t)

function (Kr::Regulator)(θ, α, μ, t)
    Pr = Kr.Pr(t)
    Rr = R(θ,α,μ,t)
    Br = fu(θ,α,μ,t)

    Rr\Br'Pr
end


nx(Kr::Regulator) = nx(Kr.θ)
nu(Kr::Regulator) = nu(Kr.θ)
extrema(Kr::Regulator) = extrema(Kr.Pr)
eachindex(Kr::Regulator) = OneTo(nu(Kr)*nx(Kr))
show(io::IO, Kr::Regulator) = println(io, preview(Kr))




# regulator(θ,φ,τ) = regulator(θ, φ.x, φ.u, τ)
# design the regulator, solving dPr_dt
 function regulator(θ::Model{NX,NU}, φ, τ; verbosity) where {NX,NU}
    iinfo("regulator"; verbosity)
    t0,tf = τ
    #FUTURE: Pf provided by user or auto-generated as P(α,μ,θ)
    # α 
    # Pf = SMatrix{NX,NX,Float64}(I(NX) - φ.x(tf)*(φ.x(tf))')
    # Pf = SMatrix{NX,NX,Float64}(I(NX))
    α = φ.x(tf)
    μ = φ.u(tf)
    Pr = ODE(dPr_dt, Pf(θ,α,μ,tf), (tf,t0), (θ,φ))
    Regulator(θ,φ,Pr)
end



function dPr_dt(Pr,(θ,φ),t)#(M, out, θ, t, φ, Pr)
    α = φ.x(t)
    μ = φ.u(t)

    Ar = fx(θ,α,μ,t)
    Br = fu(θ,α,μ,t)
    Qr = Q(θ,α,μ,t)
    Rr = R(θ,α,μ,t)
    
    return - Ar'Pr - Pr*Ar + Pr'Br*(Rr\Br'Pr) - Qr
end


