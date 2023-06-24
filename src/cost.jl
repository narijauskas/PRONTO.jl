# ----------------------------------- cost functional ----------------------------------- #

cost(ξ,τ) = cost(ξ.θ,ξ.x,ξ.u,τ)

function cost(θ,x,u,τ)
    t0,tf = τ; h0 = SVector{1,Float64}(0)
    hf = p(θ,x(tf),u(tf),tf)
    h = hf + solve(ODEProblem(dh_dt, h0, (t0,tf), (θ,x,u)), Tsit5(); reltol=1e-7)(tf)
    return h[1]
end

dh_dt(h, (θ,x,u), t) = l(θ, x(t), u(t), t)

# ----------------------------------- cost derivatives ----------------------------------- #

#TODO: decide on appropriate names for intermediate variables!
function cost_derivs(θ,λ,φ,ξ,ζ,τ; verbosity)
    iinfo("cost/derivs"; verbosity)
    t0,tf = τ

    🐱_f = solve(ODEProblem(d🐱_dt, 0, (t0,tf), (θ,ξ,ζ)), Tsit5(); reltol=1e-7)(tf)
    🐶_f = solve(ODEProblem(d🐶_dt, 0, (t0,tf), (θ,λ,ξ,ζ)), Tsit5(); reltol=1e-7)(tf)

    zf = ζ.x(tf)
    αf = φ.x(tf)
    μf = φ.u(tf)
    rf = px(θ,αf,μf,tf)
    Pf = pxx(θ,αf,μf,tf)
    Dh = 🐱_f + rf'zf
    D2g = 🐶_f + zf'Pf*zf
    return Dh,D2g
end

function d🐱_dt(🐱, (θ,ξ,ζ), t)
    x = ξ.x(t)
    u = ξ.u(t)
    z = ζ.x(t)
    v = ζ.u(t)
    a = lx(θ,x,u,t)
    b = lu(θ,x,u,t)
    return a'z + b'v
end

function d🐶_dt(🐶, (θ,λ,ξ,ζ), t)
    x = ξ.x(t)
    u = ξ.u(t)
    z = ζ.x(t)
    v = ζ.u(t)
    λ = λ(t)
    Q = Lxx(θ,λ,x,u,t)
    S = Lxu(θ,λ,x,u,t)
    R = Luu(θ,λ,x,u,t)
    return z'Q*z + 2*z'S*v + v'R*v
end
