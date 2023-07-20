
function dλ_dt(λ, (θ,ξ,Kr), t)
    x = ξ.x(t); u = ξ.u(t); Kr = Kr(t)

    A = fx(θ,x,u,t)
    B = fu(θ,x,u,t)
    a = lx(θ,x,u,t)
    b = lu(θ,x,u,t)

    -(A - B*Kr)'λ - a + Kr'b
end

# for convenience:
function lagrangian(θ,ξ,Kr,τ; verbosity)
    iinfo("lagrangian"; verbosity)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)

    λf = mx(θ, αf, μf, tf)
    λ = ODE(dλ_dt, λf, (tf,t0), (θ,ξ,Kr))
    return λ
end


abstract type SearchOrder end
struct FirstOrder <: SearchOrder end
struct SecondOrder <: SearchOrder end


struct Optimizer{Tθ,Tλ,Tξ,TP}
    N::SearchOrder
    θ::Tθ
    λ::Tλ
    ξ::Tξ
    P::TP
end

(Ko::Optimizer)(t) = Ko(Ko.ξ.x(t), Ko.ξ.u(t), t)
(Ko::Optimizer)(x,u,t) = Ko(Ko.θ,x,u,t)

function (Ko::Optimizer)(θ,x,u,t)

    λ = is2ndorder(Ko) ? Ko.λ(t) : nothing
    Q = is2ndorder(Ko) ? Lxx(θ,λ,x,u,t) : lxx(θ,x,u,t)
    S = is2ndorder(Ko) ? Lxu(θ,λ,x,u,t) : lxu(θ,x,u,t)
    R = is2ndorder(Ko) ? Luu(θ,λ,x,u,t) : luu(θ,x,u,t)

    P = Ko.P(t)
    B = fu(θ,x,u,t)

    return R\(S' + B'P)
end

order(Ko::Optimizer) = Ko.N
nx(Ko::Optimizer) = nx(Ko.θ)
nu(Ko::Optimizer) = nu(Ko.θ)
extrema(Ko::Optimizer) = extrema(Ko.P)
eachindex(Ko::Optimizer) = OneTo(nu(Ko)*nx(Ko))
show(io::IO, Ko::Optimizer) = println(io, preview(Ko))


function asymmetry(A)
    (m,n) = size(A)
    @assert m == n "must be square matrix"
    sum([0.5*abs(A[i,j]-A[j,i]) for i in 1:n, j in 1:n])
end

retcode(x::ODE) = x.soln.retcode
isstable(x::ODE) = retcode(x) == ReturnCode.Success

struct InstabilityException <: Exception
end


function optimizer(θ,λ,ξ,τ; verbosity)
    iinfo("optimizer"; verbosity)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)
    
    Pf = mxx(θ,αf,μf,tf)

    #FIX: this implementation is not the most robust
    P,N = try
        N = SecondOrder()
        P = ODE(dP_dt, Pf, (tf,t0), (θ,λ,ξ,N), verbose=false)
        !isstable(P) && throw(InstabilityException())
        (P,N)
    catch e
        N = FirstOrder()
        P = ODE(dP_dt, Pf, (tf,t0), (θ,λ,ξ,N))
        (P,N)
    end

    Ko = Optimizer(N,θ,λ,ξ,P)
    iinfo("using $(is2ndorder(Ko) ? "2nd" : "1st") order search"; verbosity)
    return Ko
end

# for debugging - only first order descent
function optimizer1(θ,λ,ξ,τ)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)
    
    Pf = mxx(θ,αf,μf,tf)
    N = FirstOrder()
    P = ODE(dP_dt, Pf, (tf,t0), (θ,λ,ξ,N))
    
    return Optimizer(N,θ,λ,ξ,P)
end

function dP_dt(P, (θ,λ,ξ,N), t)
    x = ξ.x(t); u = ξ.u(t)

    A = fx(θ,x,u,t)
    B = fu(θ,x,u,t)

    λ = is2ndorder(N) ? λ(t) : nothing
    Q = is2ndorder(N) ? Lxx(θ,λ,x,u,t) : lxx(θ,x,u,t)
    S = is2ndorder(N) ? Lxu(θ,λ,x,u,t) : lxu(θ,x,u,t)
    R = is2ndorder(N) ? Luu(θ,λ,x,u,t) : luu(θ,x,u,t)

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
(vo::Costate)(x,u,t) = vo(vo.θ,x,u,t)

function (vo::Costate)(θ,x,u,t)

    R = is2ndorder(vo) ? Luu(θ,vo.λ(t),x,u,t) : luu(θ,x,u,t)
    B = fu(θ,x,u,t)
    b = lu(θ,x,u,t)
    r = vo.r(t)

    return -R\(B'r + b)
end


function dr_dt(r, (θ,λ,ξ,Ko), t)
    x = ξ.x(t); u = ξ.u(t)
    Ko = Ko(x,u,t)

    A = fx(θ,x,u,t)
    B = fu(θ,x,u,t)
    a = lx(θ,x,u,t)
    b = lu(θ,x,u,t)

    -(A - B*Ko)'r - a + Ko'b
end

function costate(θ,λ,ξ,Ko,τ; verbosity)
    iinfo("costate"; verbosity)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)
    N = order(Ko)
    rf = mx(θ,αf,μf,tf)
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



function search_direction(θ::Model{NX,NU},ξ,Ko,vo,τ; verbosity=1, dt=0.001) where {NX,NU}
    iinfo("search_direction"; verbosity)
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
    A = fx(θ,x,u,t)
    B = fu(θ,x,u,t)
    v = vo(x,u,t) - Ko(x,u,t)*z
    return A*z + B*v
end
