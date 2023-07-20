
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
show(io::IO, Kr::Regulator) = println(io, make_plot(t->vec(Kr(t)), t_plot(Kr)))


regulator(θ, α, μ, τ; kw...) = regulator(θ, Trajectory(θ, α, μ), τ; kw...)
# design the regulator, solving dPr_dt
function regulator(θ::Model{NX,NU}, φ, τ; verbosity=0) where {NX,NU}
    iinfo("regulator"; verbosity)
    t0,tf = τ
    αf = φ.x(tf)
    μf = φ.u(tf)
    Pr = ODE(dPr_dt, Pf(θ,αf,μf,tf), (tf,t0), (θ,φ))
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


