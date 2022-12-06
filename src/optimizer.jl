
function dλ_dt(λ, (θ,ξ,Kr), t)
    x = ξ.x(t); u = ξ.u(t); Kr = Kr(t)

    A = fx(x,u,t,θ)
    B = fu(x,u,t,θ)
    a = lx(x,u,t,θ)
    b = lu(x,u,t,θ)

    -(A - B*Kr)'λ - a + Kr'b
end

# for convenience:
function lagrangian(θ,ξ,φ,Kr,τ)
    t0,tf = τ
    αf = φ.x(tf)
    μf = φ.u(tf)

    λf = px(αf, μf, tf, θ)
    λ = ODE(dλ_dt, λf, (tf,t0), (θ,ξ,Kr))
    return λ
end







abstract type SearchOrder end
struct FirstOrder <: SearchOrder end
struct SecondOrder <: SearchOrder end





# M,N,Λ,Ξ,P
struct Optimizer{Tθ,Tλ,Tξ,TP}
    N::SearchOrder
    θ::Tθ
    λ::Tλ
    ξ::Tξ
    P::TP
end

(Ko::Optimizer)(t) = Ko(Ko.ξ.x(t), Ko.ξ.u(t), t)
(Ko::Optimizer)(x,u,t) = Ko(x,u,t,Ko.θ)

function (Ko::Optimizer)(x,u,t,θ)

    λ = is2ndorder(Ko) ? Ko.λ(t) : nothing
    Q = is2ndorder(Ko) ? Lxx(λ,x,u,t,θ) : lxx(x,u,t,θ)
    S = is2ndorder(Ko) ? Lxu(λ,x,u,t,θ) : lxu(x,u,t,θ)
    R = is2ndorder(Ko) ? Luu(λ,x,u,t,θ) : luu(x,u,t,θ)

    P = Ko.P(t)
    B = fu(x,u,t,θ)

    return R\(S' + B'P)

    # return (Ko,vo)
end

order(Ko::Optimizer) = Ko.N
nx(Ko::Optimizer) = nx(Ko.θ)
nu(Ko::Optimizer) = nu(Ko.θ)
extrema(Ko::Optimizer) = extrema(Ko.P)
eachindex(Ko::Optimizer) = OneTo(nu(Ko)*nx(Ko))
show(io::IO, Ko::Optimizer) = println(io, preview(Ko))

# R = isapprox(vo) ? L(...) : l(...)


function asymmetry(A)
    (m,n) = size(A)
    @assert m == n "must be square matrix"
    sum([0.5*abs(A[i,j]-A[j,i]) for i in 1:n, j in 1:n])
end


# ro_f = collect(px(M,θ,tf,φ(tf)))


retcode(x::ODE) = x.soln.retcode
isstable(x::ODE) = :Success == retcode(x)


function optimizer(θ,λ,ξ,φ,τ)
    t0,tf = τ
    αf = φ.x(tf)
    μf = φ.u(tf)
    
    Pf = pxx(αf,μf,tf,θ)
    N = SecondOrder()
    P2 = ODE(dP_dt, Pf, (tf,t0), (θ,λ,ξ,N); verbose=false)

    if isstable(P2)
        P = P2
    else
        N = FirstOrder()
        P = ODE(dP_dt, Pf, (tf,t0), (θ,λ,ξ,N))
    end
    return Optimizer(N,θ,λ,ξ,P)
end

function dP_dt(P, (θ,λ,ξ,N), t)
    x = ξ.x(t); u = ξ.u(t)

    A = fx(x,u,t,θ)
    B = fu(x,u,t,θ)

    λ = is2ndorder(N) ? λ(t) : nothing
    Q = is2ndorder(N) ? Lxx(λ,x,u,t,θ) : lxx(x,u,t,θ)
    S = is2ndorder(N) ? Lxu(λ,x,u,t,θ) : lxu(x,u,t,θ)
    R = is2ndorder(N) ? Luu(λ,x,u,t,θ) : luu(x,u,t,θ)

    Ko = R\(S' + B'P)
    return - A'P - P*A + Ko'R*Ko - Q
end










struct Costate{Tθ,Tλ,Tξ,Tr}
    N::SearchOrder
    θ::Tθ
    λ::Tλ # ODE{SVector{NX,Float64}}
    ξ::Tξ
    r::Tr
end


(vo::Costate)(t) = vo(vo.ξ.x(t), vo.ξ.u(t), t)
(vo::Costate)(x,u,t) = vo(x,u,t,vo.θ)

function (vo::Costate)(x,u,t,θ)

    R = is2ndorder(vo) ? Luu(vo.λ(t),x,u,t,θ) : luu(x,u,t,θ)
    B = fu(x,u,t,θ)
    b = lu(x,u,t,θ)
    r = vo.r(t)

    return -R\(B'r + b)
end




function dr_dt(r, (θ,λ,ξ,Ko), t)
    x = ξ.x(t); u = ξ.u(t)
    Ko = Ko(x,u,t)

    A = fx(x,u,t,θ)
    B = fu(x,u,t,θ)
    a = lx(x,u,t,θ)
    b = lu(x,u,t,θ)

    -(A - B*Ko)'r - a + Ko'b
end

function costate(θ,λ,ξ,φ,Ko,τ)
    t0,tf = τ
    αf = φ.x(tf)
    μf = φ.u(tf)
    N = order(Ko)
    rf = px(αf,μf,tf,θ)
    r = ODE(dr_dt, rf, (tf,t0), (θ,λ,ξ,Ko))
    return Costate(N,θ,λ,ξ,r)
end


order(vo::Costate) = vo.N
nx(vo::Costate) = nx(vo.θ)
nu(vo::Costate) = nu(vo.θ)
extrema(vo::Costate) = extrema(vo.r)
eachindex(vo::Costate) = OneTo(nu(vo)^2)
show(io::IO, vo::Costate) = println(io, preview(vo))


is2ndorder(Ko::Optimizer) = is2ndorder(Ko.N)
is2ndorder(vo::Costate) = is2ndorder(vo.N)
is2ndorder(::SecondOrder) = true
is2ndorder(::Any) = false



function search_direction(θ::Model{NX,NU},ξ,Ko,vo,τ; dt=0.001) where {NX,NU}
    t0,tf = τ
    ts = t0:dt:tf
    vbuf = Vector{SVector{NU,Float64}}()

    cb = FunctionCallingCallback(funcat = ts) do z,t,integrator
        (_,ξ,Ko,vo) = integrator.p

        x = ξ.x(t)
        u = ξ.u(t)
        v = vo(x,u,t) - Ko(x,u,t)*z

        push!(vbuf, SVector{NU,Float64}(v))
    end
    z0 = zeros(SVector{NX,Float64})
    z = ODE(dz_dt, z0, (t0,tf), (θ,ξ,Ko,vo); callback = cb, saveat = ts)
    v = Interpolant(scale(interpolate(vbuf, BSpline(Linear())), ts))

    return Trajectory(θ,z,v)
end

function dz_dt(z, (θ,ξ,Ko,vo), t)
    x = ξ.x(t)
    u = ξ.u(t)
    A = fx(x,u,t,θ)
    B = fu(x,u,t,θ)
    v = vo(x,u,t) - Ko(x,u,t)*z
    return A*z + B*v
end


# @build $T dy_dt(M,θ,t,ξ,ζ,λ) -> vcat(

#     ($a)'*($z) + ($b)'*($v),
#     ($z)'*($Qo_2)*($z) + 2*($z)'*($So_2)*($v) + ($v)'*($Ro_2)*($v)
# )
# @build $T _Dh(M,θ,t,φ,ζ,y) -> y[1] + (PRONTO.px($M,θ,t,φ))'*($z)
# @build $T _D2g(M,θ,t,φ,ζ,y) -> y[2] + ($z)'*PRONTO.pxx($M,θ,t,φ)*($z)
# @tock; println(@clock)
