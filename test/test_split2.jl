nothing
##
using StaticArrays, LinearAlgebra, Symbolics
using SparseArrays, MatrixEquations
using MacroTools, BenchmarkTools

using PRONTO

# NX = 22; NU = 1; NΘ = 0
# struct Split <: Model{NX,NU,NΘ}
# end

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

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where those encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.kr == θ[1] and θ.kq == θ[2]


# ------------------------------- model def ------------------------------- ##
# for now:
NX = 22; NU = 1

struct SplitP <: Model{22,1,3}
    α::Float64
    kr::Float64
    kq::Float64
end


function dynamics(x,u,t,θ)
    ω = 1.0
    n = 5
    α = θ[1]
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H2 = v*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    return mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end



# function regulator(x,u,t,θ)
#     R = θ.kr*I(NU)
#     Q = θ.kq*I(NX)
#     return Q,R
# end

Rreg(x,u,t,θ) = θ[2]*I(NU)
Qreg(x,u,t,θ) = θ[3]*I(NX)

stagecost(x,u,t,θ) = 1/2 * collect(u')I*u

# get the ith eigenstate
function x_eig(i)
    n = 5
    α = 10 
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,i])
end

function termcost(x,u,t,θ)
    P = I(NX) - inprod(x_eig(2))
    1/2 * collect(x')*P*x
end


PRONTO.generate_model(SplitP, dynamics, stagecost, termcost, Qreg, Rreg)
## ------------------------------- symbolic derivatives ------------------------------- ##

    


## ------------------------------- testing ------------------------------- ##
NX = 22; NU = 1

x0 = SVector{NX}(x_eig(1))
xf = SVector{NX}(x_eig(2))

u0 = 0.0
t0,tf = τ = (0,10)

θ = SplitP(10,1,1)
μ = @closure t->SizedVector{1}(u0)

φ = open_loop(θ,xf,μ,τ)
pronto(θ,x0,φ,τ)

Kr = regulator(θ,φ,τ)
ξ = projection(θ,x0,φ,Kr,τ)

λ = PRONTO.lagrangian(θ,ξ,φ,Kr,τ)
Ko = PRONTO.optimizer(θ,λ,ξ,φ,τ)
vo = PRONTO.costate(θ,λ,ξ,φ,Ko,τ)


ζ = PRONTO.search_direction(θ,ξ,Ko,vo,τ)
γ = 0.7
ξ1 = PRONTO.armijo_projection(θ,x0,ξ,ζ,γ,Kr,τ)

nothing
## ------------------------------- testing ------------------------------- ##



α = ODE(dx_dt_ol!, xf, (t0,tf), (θ,μ), Size(xf))
x = ODE(dx_dt_ol!, x0, (t0,tf), (θ,μ), Size(x0))
# u = @closure t->u_ol(θ,μ,t)

Prf = SizedMatrix{NX,NX}(diagm(rand(0.9:0.001:1.1,NX)))

Pr = ODE(dPr_dt!, Prf, (tf,t0), (θ,α,μ), Size(NX,NX))

x = ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))


# dx = similar(collect(x0))
# dx_dt!(dx,x0,(θ,μ(t0)),t0)
using SciMLBase, DifferentialEquations
Base.show(io::IO, ::Type{<:SciMLBase.ODEProblem}) = print(io, "ODEProblem{...}")
Base.show(io::IO, ::Type{<:OrdinaryDiffEq.ODEIntegrator}) = print(io, "ODEIntegrator{...}")



## ------------------------------- test regulator ------------------------------- ##

function test_Kr1(α,μ,Pr,θ)
    t = rand(0.:10.)
    PRONTO.Kr(α(t),μ(t),Pr(t),t,θ)
end


function test_Kr2(K)
    t = rand(0.:10.)
    K(t)
end



test1(fn) = fn(rand(0.:10.))

@benchmark test1($α)
@benchmark test1($μ)
@benchmark test1($Pr)

test_Kr1(α,μ,Pr,θ)
@time test_Kr1(α,μ,Pr,θ)
@benchmark test_Kr1($α,$μ,$Pr,$θ)


t = rand(0.:10.)

@benchmark Kr2 = Regulator(θ,α,μ,Pr)


test_Kr2(Kr2)
@time test_Kr2(Kr2)
@benchmark test_Kr2($Kr2)


@btime α(1)


Pf = SMatrix{NX,NX,Float64}(I(NX))

@benchmark ODE(dPr_dt!, Pf, (T.tf,T.t0), (θ,α,μ), Size(Pf))

T = TimeDomain(t0,tf)
@benchmark Regulator(θ,T,α,μ)




struct Projection{M,T1} <: Timeseries
    θ
    T
    x
    u #closure holding α,μ,Kr
    # or directly:
    # α
    # μ
    # Kr
end

ξ.x
ξ.u


u0 = μ(0)-Kr0(0)*(x0-α(0))
T = TimeDomain(0,10)
Kr = Regulator(θ,T,α,μ)
@benchmark Regulator(θ,T,α,μ) # 88 ms
@benchmark Kr(rand()) # 3.3 μs ~ 5μs
x = ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr), Size(x0))
@benchmark ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr), Size(x0)) # 19 ms
@benchmark x(rand()) # 839 ns ~ 2μs
@benchmark ODE(dae, ξ0, (t0,tf), (θ,α,μ,Kr), Size(ξ0)) # 232 ms # 10x worse!


u = t -> (μ(t) - Kr(t)*(x(t)-α(t)))
@benchmark u(rand()) # 4.8 μs ~ 8.8 μs
@benchmark [u(t) for t in ts] # 81 ms
@benchmark [x(t) for t in ts] # 17 ms
us = [u(t) for t in ts] # would be saved values from ode... hopefully cheap
@benchmark linear_interpolation(ts, us) # 11 μs

massmatrix(M) = cat(diagm(ones(nx(M))), zeros(nu(M)); dims=(1,2))
dae = ODEFunction(dξ_dt!, mass_matrix = massmatrix(θ))
ξ = ODE(dae, ξ0, (t0,tf), (θ,timed_α,timed_μ,Kr0), Size(ξ0))
try
    second_order(...,λ)
catch e
    if e isa InstabilityException
        first_order(...,λ)
    else
        rethrow(e)
    end
end
using FunctionWrappers: FunctionWrapper

using TimerOutputs
const clk = TimerOutput()

a(t) = @timeit clk "α" α(t)

timed_α = clk(α, "α")
timed_μ = clk(μ, "μ")

v = [timed_α(t) for t in t]
@time QuadraticSpline(v,t);

u_fxn(x,α,μ,Kr,t) = u_fxn(x(t),α(t),μ(t),Kr(t))
u_fxn(x,α,μ,Kr) = μ - Kr*(x-α)
@btime u_fxn(x,α,μ,Kr,t0)

# @btime begin
    Kr0 = Regulator(θ,T,α,μ);
    x1 = ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr0), Size(x0));
    # u1 = t -> (μ(t) - Kr0(t)*(x1(t)-α(t)))
    u1 = cubic_spline_interpolation(ts, [u_fxn(x1,α,μ,Kr0,t) for t in ts]);
# @time begin
    Kr1 = Regulator(θ,T,x1,u1);
    x2 = ODE(dx_dt2!, x0, (t0,tf), (θ,x1,u1,Kr1), Size(x0));
    # u2 = t -> (u1(t) - Kr1(t)*(x2(t)-x1(t)))
    u2 = linear_interpolation(ts, [u_fxn(x2,x1,u1,Kr1,t) for t in ts]);

    Kr2 = Regulator(θ,T,x2,u2);
    x3 = ODE(dx_dt2!, x0, (t0,tf), (θ,x2,u2,Kr1), Size(x0));
    # u3 = t -> (u2(t) - Kr2(t)*(x3(t)-x2(t)))
# end

reset_timer!(clk)
u3(1)
clk

Base.length(::var"#21#22") = 1
# φ = Guess(x,[u]) # arbitrary curve
# Timeseries
# AbstractTimeseries

# Projection,OLTrajectory,SearchDirection <: Trajectory

φ = OLTrajectory(x0,[u]) # open loop
Kr = Regulator(φ)
# Regulator(α,μ)

Projection(φ) = Projection(φ, Regulator(φ))
Projection(α,μ) = Projection(α, μ, Regulator(α,μ))

ξ = Projection(φ, Kr::Regulator) # closed loop -> return trajectory object
ξ = Projection(α, μ, Kr::Regulator) # closed loop around arbitrary curve

ζ = SearchDirection(ξ)
    # FirstOrderSearch()
    # SecondOrderSearch()
    # Ko = Optimizer()

# armijo:
    # create candidate trajectory
    η = Projection(ξ,γ,ζ,Kr) # closed loop armijo
    # evaluate cost
    u = 

φ = η


# search direction
λ = ODE(...)
P = ODE(...)
r = ODE(...)





Pf = SMatrix{NX,NX,Float64}(I(NX))
Pr = ODE(dPr_dt!, Pf, (T.tf,T.t0), (θ,α,μ), Size(Pf))

@benchmark ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))

Kr = Regulator(θ,T,α,μ)

@benchmark ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr), Size(x0))


dt = 0.001
using DiffEqCallbacks
const XBUF = Vector{SVector{22,Float64}}()
const xbuf = Vector{MVector{22,Float64}}()

for _ in TT
    push!(xbuf, zeros(MVector{22}))
end

const TT = t0:dt:tf

@benchmark ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr), Size(x0); callback = cb)


@btime [zeros(MVector{NX+NU}) for _ in TT] # 9ms
@btime map(t->zeros(MVector{NX+NU}), TT) # 9ms

@btime begin
    buf = Vector{MVector{22,Float64}}()
    for _ in TT
        push!(buf, zeros(MVector{22}))
    end
    return buf
end

cb = FunctionCallingCallback(update_ξ;
        funcat=TT,
        func_start = false)

        ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr), Size(x0); callback = cb)

φ = (α=α,μ=μ)




x0 = SVector{NX}(x_eig(1))
@benchmark solve(ODEProblem(dxdt, x0, (t0,tf), (θ,α,μ,Kr)),Tsit5(); reltol=1e-7, callback = cb)
@benchmark solve(ODEProblem(dxdt!, collect(x0), (t0,tf), (θ,α,μ,Kr)),Tsit5(); reltol=1e-7, callback = cb)

function dxdt!(dx,x,(θ,α,μ,Kr),t)
    α = α(t)
    μ = μ(t)
    Kr = Kr(α,μ,t)
    u = μ - Kr*(x-α)
    PRONTO.f!(dx,x,u,t,θ)
end
ODE(dx_dt2!, x0, (t0,tf), (θ,α,μ,Kr), Size(x0); callback = cb)


@code_warntype scale(interpolate(XBUF, BSpline(Linear())), TT)


nothing




























































PRONTO.fx(α(t0),μ(t0),t0,θ)

PRONTO.Kr(α(tf),μ(tf),Prf,tf,θ)

dPr = zeros(NX,NX)
dPr_dt!(dPr,collect(Prf),(θ,α,μ),t0)

PRONTO.Q(α(tf),μ(tf),tf,θ)


# this is insanely cool:
μ = @closure t->SizedVector{1}(0.1)
ODE(dx_dt_ol!, x0, (t0,tf), (θ,μ), Size(x0))
for x0 in x_eig.(1:11)
    show(ODE(dx_dt_ol!, x0, (t0,tf), (θ,μ), Size(x0)))
end






tsk = map(1:100) do kr
    θ = SplitP(kr, 1)
    @spawn ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))
end

fetch.(tsk)





θ = SplitP(1, 1)
Pr = @closure t->Prf
x = ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))














# trace symbolic
    # option 1. give symbolics and intermediate constructors to build a model struct
    # option 2. model macro
    # option 3. user passes functions to central constructor
# build symbolic
# build expression
# cleanup
# load into PRONTO






# how to trace these? reverse lookup on type?
# Rr(θ,x,u,t) = θ.kr*I(nu(θ))
# Qr(θ,x,u,t) = θ.kq*I(nx(θ))


# @define SplitP begin
    



# end


# import LinearAlgebra: Tridiagonal, SymTridiagonal
# SymTridiagonal(dv,ev) = SymTridiagonal(promote(dv,ev)...)
# Tridiagonal(dl,d,du) = Tridiagonal(promote(dl,d,du)...)

# @variables θ[1:NΘ] x[1:NX] u[1:NU]
# α = θ[3]; ω = θ[4]


function ye()
    tmp = MVector{22,Float64}(undef)
    fill!(tmp, 0)
    return SVector(tmp)
end

function bench1()
    tmp = MMatrix{22,22,Float64}(undef)
    @. tmp = rand()
    return SMatrix(tmp)
end

# approx breakdown
rand()*484 # 212 ns

#232 ns
function yee()
    tmp = MMatrix{22,22,Float64}(undef)
    for i in eachindex(tmp)
        tmp[i] = i
    end
    return SMatrix(tmp)
end

@btime yee() # 232 ns
@allocated yee() # 0 allocs


nu = eigvecs(H0)

nu1 = nu[:,1]
nu2 = nu[:,2]
# T = :Split
# ex = quote

    mdl = @model Split begin
        using LinearAlgebra
            
        NX = 22; NU = 1; NΘ = 0
    
        function mprod(x)
            Re = I(2)  
            Im = [0 -1;
                1 0]   
            M = kron(Re,real(x)) + kron(Im,imag(x));
            return M   
        end
    
        function inprod(x)
            a = x[1:Int(NX/2)]
            b = x[(Int(NX/2)+1):(2*Int(NX/2))]
            P = [a*a'+b*b' -(a*b'+b*a');
                a*b'+b*a' a*a'+b*b']
            return P
        end
    
        N = 5
        n = 2*N+1
        α = 10
        ω = 0.5
    
        H = zeros(n,n)
        for i = 1:n
            H[i,i] = 4*(i-N-1)^2
        end
        v = -α/4 * ones(n-1)
    
        H0 = H + Bidiagonal(zeros(n), v, :U) + Bidiagonal(zeros(n), v, :L)
        H1 = Bidiagonal(zeros(n), -v*1im, :U) + Bidiagonal(zeros(n), v*1im, :L)
        H2 = Bidiagonal(zeros(n), -v, :U) + Bidiagonal(zeros(n), -v, :L)
    
        nu = eigvecs(H0)
    
        nu1 = nu[:,1]
        nu2 = nu[:,2]
    
        f(θ,t,x,u) = mprod(-1im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2))*x
        
        Ql = zeros(2*n,2*n)
        Rl = I
        l(θ,t,x,u) = 1/2*x'*Ql*x + 1/2*u'*Rl*u
        
        Rr(θ,t,x,u) = diagm(ones(1))
        Qr(θ,t,x,u) = diagm(ones(2*n))
        
        xf = [nu2;0*nu2]
        function p(θ,t,x,u)
            P = I(2*n) - inprod(xf)
            1/2*x'*P*x
        end
    
    
    end




# NX = 22; NU = 1; NΘ = 4
# struct SplitP <: Model{NX,NU,NΘ}
#     kr::Float64
#     kq::Float64
#     α::Float64 # 10
#     ω::Float64 # 0.5
# end