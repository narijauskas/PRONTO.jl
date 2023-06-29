using PRONTO
using StaticArrays, LinearAlgebra

N = 4
NX = 2*(2*N+1)
NU = 1

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

function inprod(x)
    i = Int(length(x)/2)
    a = x[1:i]
    b = x[i+1:end]
    P = [a*a'+b*b' -(a*b'+b*a');
        a*b'+b*a' a*a'+b*b']
    return P
end

# get the ith eigenstate
function x_eig(i)
    α = 10 
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where integer type parameters {NX,NU,NΘ} encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr ≡ θ[2] and θ.kq ≡ θ[3]


## ------------------------------- beam splitter to eigenstate 2 ------------------------------- ##

@kwdef struct bs2 <: PRONTO.Model{NX,NU}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


@dynamics bs2 begin
    α = 10 
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -N:N], v*ones(2N))...)
    H1 = v*im*Tridiagonal(ones(2N), zeros(2N+1), -ones(2N))
    H2 = v*Tridiagonal(-ones(2N), zeros(2N+1), -ones(2N))
    return mprod(-im*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end

@stage_cost bs2 begin
    θ.kl/2*u'*I*u 
end

@terminal_cost bs2 begin
    P = I(NX) - inprod(x_eig(2))
    return 1/2*x'*P*x
end

@regulatorQ bs2 begin
    x_re = x[1:2N+1]
    x_im = x[2N+2:NX]
    ψ = x_re + im*x_im
    return θ.kq*mprod(I(2N+1) - ψ*ψ')
end

@regulatorR bs2 θ.kr*I(NU)

# must be run after any changes to model definition
resolve_model(bs2)

# overwrite default behavior of Pf
PRONTO.Pf(θ::bs2,α,μ,tf) = SMatrix{NX,NX,Float64}(I(NX)-α*α')

# runtime plots
PRONTO.runtime_info(θ::bs2, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))


## ------------------------------- demo: eigenstate 1->2 in 10s ------------------------------- ##


x0 = SVector{NX}(x_eig(1))
xf = SVector{NX}(x_eig(2))
t0,tf = τ = (0,10)


θ = bs2(kl=0.01, kr=1, kq=1)
μ = t->SVector{NU}(0.4*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 50, limitγ = true, verbose=1)



## ------------------------------- demo: eigenstate 1->4 in 10s ------------------------------- ##

x0 = SVector{22}(x_eig(1))
xf = SVector{22}(x_eig(4))
t0,tf = τ = (0,10)


θ = Split4(kl=0.01, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)

# plot_split(ξ,τ)


## ------------------------------- demo: eigenstate 1->4 in 2s ------------------------------- ##

x0 = SVector{22}(x_eig(1))
xf = SVector{22}(x_eig(4))
t0,tf = τ = (0,2.55)


θ = Split4(kl=0.02, kr=1, kq=1)
μ = @closure t->SVector{1}(0.5*sin(t))
φ = open_loop(θ,x0,μ,τ)
ξ = pronto(θ,x0,φ,τ; tol = 1e-6, maxiters = 100, limitγ = true)

plot_split(ξ,τ)








## ------------------------------- step-by-step debugging ------------------------------- ##

Kr = PRONTO.regulator(θ,φ,τ)
ξ = PRONTO.projection(θ,x0,φ,Kr,τ)

λ = PRONTO.lagrangian(θ,ξ,φ,Kr,τ)
Ko = PRONTO.optimizer(θ,λ,ξ,φ,τ)
vo = PRONTO.costate(θ,λ,ξ,φ,Ko,τ)


ζ = PRONTO.search_direction(θ,ξ,Ko,vo,τ)
γ = 0.7
ξ1 = PRONTO.armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)


## ------------------------------- other ------------------------------- ##


# shorter error messages:
using SciMLBase, DifferentialEquations
Base.show(io::IO, ::Type{<:SciMLBase.ODEProblem}) = print(io, "ODEProblem{...}")
Base.show(io::IO, ::Type{<:OrdinaryDiffEq.ODEIntegrator}) = print(io, "ODEIntegrator{...}")


