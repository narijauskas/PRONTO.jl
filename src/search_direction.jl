
# ----------------------------------- lagrange multipliers ----------------------------------- #

function dλ_dt(λ, (θ,ξ,Kr), t)
    x = ξ.x(t); u = ξ.u(t); Kr = Kr(t)

    A = fx(θ,x,u,t)
    B = fu(θ,x,u,t)
    a = lx(θ,x,u,t)
    b = lu(θ,x,u,t)

    -(A - B*Kr)'λ - a + Kr'b
end

# for convenience:
function lagrange(θ,ξ,Kr,τ)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)

    λf = mx(θ, αf, μf, tf)
    λ = ODE(dλ_dt, λf, (tf,t0), (θ,ξ,Kr))
    return λ
end


# ----------------------------------- optimizer ----------------------------------- #

abstract type SearchOrder end
struct FirstOrder <: SearchOrder end
struct SecondOrder <: SearchOrder end


struct OptFBGain{Tθ,Tλ,Tξ,TP}
    N::SearchOrder
    θ::Tθ
    λ::Tλ
    ξ::Tξ
    P::TP
end

(Ko::OptFBGain)(t) = Ko(Ko.ξ.x(t), Ko.ξ.u(t), t)
(Ko::OptFBGain)(x,u,t) = Ko(Ko.θ,x,u,t)

function (Ko::OptFBGain)(θ,x,u,t)

    λ = is2ndorder(Ko) ? Ko.λ(t) : nothing
    Q = is2ndorder(Ko) ? Lxx(θ,λ,x,u,t) : lxx(θ,x,u,t)
    S = is2ndorder(Ko) ? Lxu(θ,λ,x,u,t) : lxu(θ,x,u,t)
    R = is2ndorder(Ko) ? Luu(θ,λ,x,u,t) : luu(θ,x,u,t)

    P = Ko.P(t)
    B = fu(θ,x,u,t)

    return R\(S' + B'P)
end

order(Ko::OptFBGain) = Ko.N
nx(Ko::OptFBGain) = nx(Ko.θ)
nu(Ko::OptFBGain) = nu(Ko.θ)
domain(Ko::OptFBGain) = domain(Ko.P)
eachindex(Ko::OptFBGain) = OneTo(nu(Ko)*nx(Ko))
show(io::IO, Ko::OptFBGain) = println(io, println(io, make_plot(t->vec(Ko(t)), t_plot(Ko))))


function asymmetry(A)
    (m,n) = size(A)
    @assert m == n "must be square matrix"
    sum([0.5*abs(A[i,j]-A[j,i]) for i in 1:n, j in 1:n])
end

retcode(x::ODE) = x.retcode
isstable(x::ODE) = retcode(x) == ReturnCode.Success

struct InstabilityException <: Exception
end


function opt_fb_gain(θ,λ,ξ,τ)
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

    Ko = OptFBGain(N,θ,λ,ξ,P)
    return Ko
end

# for debugging - only first order descent
function opt_fb_gain_1(θ,λ,ξ,τ)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)
    
    Pf = mxx(θ,αf,μf,tf)
    N = FirstOrder()
    P = ODE(dP_dt, Pf, (tf,t0), (θ,λ,ξ,N))
    
    return OptFBGain(N,θ,λ,ξ,P)
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




# ----------------------------------- costate ----------------------------------- #

struct OptFFWInput{Tθ,Tλ,Tξ,Tr}
    N::SearchOrder
    θ::Tθ
    λ::Tλ # ODE{SVector{NX,Float64}}
    ξ::Tξ
    r::Tr
end


(vo::OptFFWInput)(t) = vo(vo.ξ.x(t), vo.ξ.u(t), t)
(vo::OptFFWInput)(x,u,t) = vo(vo.θ,x,u,t)

function (vo::OptFFWInput)(θ,x,u,t)

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

function opt_ffw_input(θ,λ,ξ,Ko,τ)
    t0,tf = τ
    αf = ξ.x(tf)
    μf = ξ.u(tf)
    N = order(Ko)
    rf = mx(θ,αf,μf,tf)
    r = ODE(dr_dt, rf, (tf,t0), (θ,λ,ξ,Ko))
    return OptFFWInput(N,θ,λ,ξ,r)
end


order(vo::OptFFWInput) = vo.N
nx(vo::OptFFWInput) = nx(vo.θ)
nu(vo::OptFFWInput) = nu(vo.θ)
domain(vo::OptFFWInput) = domain(vo.r)
eachindex(vo::OptFFWInput) = OneTo(nu(vo)^2)
show(io::IO, vo::OptFFWInput) = println(io, make_plot(t->vec(vo(t)), t_plot(vo)))


is2ndorder(Ko::OptFBGain) = is2ndorder(Ko.N)
is2ndorder(vo::OptFFWInput) = is2ndorder(vo.N)
is2ndorder(::SecondOrder) = true
is2ndorder(::Any) = false



# ----------------------------------- search direction ----------------------------------- #


function search_direction(θ::Model{NX,NU},ξ,Ko,vo,τ; resample_dt=0.001) where {NX,NU}
    t0,tf = τ
    # ts = t0:resample_dt:tf
    # vbuf = Vector{SVector{NU,Float64}}(undef, length(ts))
    # zbuf = Vector{SVector{NX,Float64}}(undef, length(ts))
    i = Ref(1)

    
    sv = SavedValues(Float64, SVector{NU,Float64})
    cb = SavingCallback(sv) do z,t,integrator
        (_,_ξ,_Ko,_vo) = integrator.p
        x = _ξ.x(t)
        u = _ξ.u(t)
        return SVector{NU,Float64}(_vo(x,u,t) - _Ko(x,u,t)*z)
    end

    z0 = zeros(SVector{NX,Float64})
    z = ODE(dz_dt, z0, (t0,tf), (θ,ξ,Ko,vo); callback = cb, tstops=LinRange(t0,tf,1000))
    v = VecInterpolant(
        MVector{NU, Float64}(undef),
        # [AkimaInterpolation([x[i] for x in sv.saveval], sv.t) for i in 1:NU],
        [CubicSpline([x[i] for x in sv.saveval], sv.t) for i in 1:NU],
    )

    # cb = FunctionCallingCallback(funcat = ts) do z,t,integrator
    #     (_,_ξ,_Ko,_vo) = integrator.p
    #     # local v
    #     x = _ξ.x(t)
    #     u = _ξ.u(t)
    #     local v = _vo(x,u,t) - _Ko(x,u,t)*z
    #     vbuf[i[]] = SVector{NU,Float64}(v)
    #     zbuf[i[]] = SVector{NX,Float64}(z)
    #     i[] += 1
    #     # push!(vbuf, SVector{NU,Float64}(v))
    #     # push!(zbuf, SVector{NX,Float64}(z))
    # end

    # z0 = zeros(SVector{NX,Float64})
    # ODE(dz_dt, z0, (t0,tf), (θ,ξ,Ko,vo); dense=false, callback = cb)
    # z = Interpolant(scale(interpolate(zbuf, BSpline(Cubic())), ts))
    # v = Interpolant(scale(interpolate(vbuf, BSpline(Cubic())), ts))


    # v = t->begin
    #     local x = ξ.x(t)
    #     local u = ξ.u(t)
    #     return vo(x,u,t) - Ko(x,u,t)*z(t)
    # end
    # v = DataInterpolations.CubicSpline(sv.saveval, sv.t)
    # v = DataInterpolations.AkimaInterpolation(first.(sv.saveval), sv.t)



    # v = VecInterpolant(
    #     MVector{NU, Float64}(undef),
    #     [interpolate(sv.t, [x[i] for x in sv.saveval], SteffenMonotonicInterpolation()) for i in 1:NU]
    # )
    

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
