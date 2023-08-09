using PRONTO
using StaticArrays
using LinearAlgebra
using Base: @kwdef


function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end

function flattop(t; T, t_rise, t₀=0.0, t_fall=t_rise, func=:blackman)
    if func == :blackman
        return flattop_blackman(t, t₀, T, t_rise, t_fall)
    elseif func == :sinsq
        return flattop_sinsq(t, t₀, T, t_rise, t_fall)
    else
        throw(
            ArgumentError("Unknown func=$func. Accepted values are :blackman and :sinsq.")
        )
    end
end


function flattop_blackman(t, t₀, T, t_rise, t_fall=t_rise)
    f::Float64 = 0.0
    if t₀ ≤ t ≤ T
        f = 1.0
        if t < t₀ + t_rise
            f = blackman(t, t₀, t₀ + 2 * t_rise)
        elseif t > T - t_fall
            f = blackman(t, T - 2 * t_fall, T)
        end
    end
    return f
end

function blackman(t, t₀, T; a=0.16)
    ΔT = T - t₀
    return (
        0.5 *
        box(t, t₀, T) *
        (1.0 - a - cos(2π * (t - t₀) / ΔT) + a * cos(4π * (t - t₀) / ΔT))
    )
end

box(t, t₀, T) = (t₀ ≤ t ≤ T) ? 1.0 : 0.0


NX = 4
NU = 1

@kwdef struct Spin2 <: PRONTO.Model{NX,NU}
    kl::Float64 = 0.01
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end


@define_f Spin2 begin
    H0 = 0.5*[1 0;0 -1]
    H1 = [0 1;1 0]
    return mprod(-im * (H0 + u[1]*H1)) * x
end


@define_l Spin2 begin
    kl/2*(u'*I*u) 
end

@define_m Spin2 begin
    P = [1 0 0 0;
         0 0 0 0;
         0 0 1 0;
         0 0 0 0]
    return 1/2*(x'*P*x)
end

@define_Q Spin2 begin
    x_re = x[1:2]
    x_im = x[3:4]
    ψ = x_re + im*x_im
    return kq*mprod(I(2) - ψ*ψ')
end

@define_R Spin2 kr*I(NU)
PRONTO.Pf(θ::Spin2,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX)-α*α')

# must be run after any changes to model definition
resolve_model(Spin2)


## ------------------------------- demo: |0⟩ -> |1⟩ in 10 ------------------------------- ##

x0 = SVector{NX}([1 0 0 0])
xf = SVector{NX}([0 1 0 0])
t0,tf = τ = (0,5)
tlist = collect(range(0, 5, length=500));

θ = Spin2()
μ = t -> [0.2 * flattop(t, T=5, t_rise=0.3, func=:blackman)];
φ = open_loop(θ,x0,μ,τ)
@time ξ,data = pronto(θ,x0,φ,τ; tol=1e-4)

# terminal cost for each iteration
[PRONTO.p(θ,ξ.x(tf),ξ.u(tf),tf) for ξ in data.ξ]*2
