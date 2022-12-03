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


function ff(x,u,t,θ)
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

Rr(x,u,t,θ) = θ[2]*I(NU)
Qr(x,u,t,θ) = θ[3]*I(NX)

ll(x,u,t,θ) = 1/2 * collect(u')I*u

# get the ith eigenstate
function x_eig(i)
    n = 5
    α = 10 
    v = -α/4
    H0 = SymTridiagonal(promote([4.0i^2 for i in -n:n], v*ones(2n))...)
    w = eigvecs(collect(H0)) # symbolic doesn't work here
    x_eig = kron([1;0],w[:,2])
end

function pp(x,u,t,θ)
    P = I(NX) - inprod(x_eig(2))
    1/2 * collect(x')*P*x
end


## ------------------------------- symbolic derivatives ------------------------------- ##
# (Jx,Ju,f,l,p,R,Q) = 
PRONTO.generate_model(SplitP,ff,ll,pp,Rr,Qr)

T = SplitP

build(Size(NX,NX), :(Q(x,u,t,θ::$T)), Q)
build(Size(NU,NU), :(R(x,u,t,θ::$T)), R)

build(InPlace(), :(f!(out,x,u,t,θ::$T)), f)

build(Size(NX), :(f(x,u,t,θ::$T)), f)
build(Size(NX,NU), :(fx(x,u,t,θ::$T)), Jx(f))
build(Size(NU,NU), :(fu(x,u,t,θ::$T)), Ju(f))

lx = reshape(Jx(l),NX)
lu = reshape(Ju(l),1)

build(Size(1), :(l(x,u,t,θ::$T)), l)
build(Size(NX,NU), :(lx(x,u,t,θ::$T)), lx)
build(Size(NU,NU), :(lu(x,u,t,θ::$T)), lu)

lxx = Jx(reshape(Jx(l),NX))
lxu = Ju(reshape(Jx(l),NX))
luu = Ju(reshape(Ju(l),1))

build(Size(NX,NX), :(lxx(x,u,t,θ::$T)), lxx)
build(Size(NX,NU), :(lxu(x,u,t,θ::$T)), lxu)
build(Size(NU,NU), :(luu(x,u,t,θ::$T)), luu)

fxx = Jx(Jx(f))
fxu = Ju(Jx(f))
fuu = Ju(Ju(f))

Lxx = lxx + sum(λ[k]*fxx[k,:,:] for k in 1:NX)
Lxu = lxu + sum(λ[k]*fxu[k,:,:] for k in 1:NX)
Luu = luu + sum(λ[k]*fuu[k,:,:] for k in 1:NX)

build(Size(NX,NX), :(Lxx(λ,x,u,t,θ::$T)), Lxx)
build(Size(NX,NU), :(Lxu(λ,x,u,t,θ::$T)), Lxu)
build(Size(NU,NU), :(Luu(λ,x,u,t,θ::$T)), Luu)


build(Size(1), :(p(x,u,t,θ::$T)), p)
build(Size(NX), :(px(x,u,t,θ::$T)), Jx(p))
build(Size(NX,NX), :(pxx(x,u,t,θ::$T)), Jx(Jx(p)))


function appendto!(filename, str)
    write(io, "hello")
end

build(Size(NX,NU), :(Lxu(λ,x,u,t,θ::$T)), Lxu)

build(Size(NX,NX), :(fx(x,u,t,θ::$T)), Jx(f_sym))


ex = :(PRONTO.fx(x,u,t,θ::SplitP) = Jx(f_sym))



    # # generate method definitions for PRONTO functions
    # iinfo("differentiating model dynamics\n")
    # build_defs!(M, :f, T, (θ, t, ξ), f)
    # build_defs!(M, :fx, T, (θ, t, ξ), Jx(f))
    # build_defs!(M, :fu, T, (θ, t, ξ), Ju(f))
    # build_defs!(M, :fxx, T, (θ, t, ξ), Jx(Jx(f)))
    # build_defs!(M, :fxu, T, (θ, t, ξ), Ju(Jx(f)))
    # build_defs!(M, :fuu, T, (θ, t, ξ), Ju(Ju(f)))

    # iinfo("differentiating stage cost\n")
    # build_defs!(M, :l, T, (θ, t, ξ), l)
    # build_defs!(M, :lx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX))
    # build_defs!(M, :lu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU))
    # build_defs!(M, :lxx, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Jx)
    # build_defs!(M, :lxu, T, (θ, t, ξ), l |> Jx |> lx->reshape(lx,NX) |> Ju)
    # build_defs!(M, :luu, T, (θ, t, ξ), l |> Ju |> lu->reshape(lu,NU) |> Ju)
    # # @build lx(θ,t,ξ)->reshape(l|>Jx,NX)

    # iinfo("differentiating terminal cost\n")
    # build_defs!(M, :p, T, (θ, t, ξ), p)
    # build_defs!(M, :px, T, (θ, t, ξ), p |> Jx |> px->reshape(px,NX))
    # build_defs!(M, :pxx, T, (θ, t, ξ), p |> Jx |> px->reshape(px,NX) |> Jx)

    # iinfo("building regulator functions\n")
    # build_defs!(M, :Qrr, T, (θ, t, ξ), Qr)
    # build_defs!(M, :Rrr, T, (θ, t, ξ), Rr)

































using PRONTO: Jacobian
using Base: invokelatest
using PRONTO: now, crispr, clean
using PRONTO: define, build_expr
using PRONTO: build_inplace


# iinfo("initializing symbolics\n")
# create symbolic variables & operators
@variables x[1:22] u[1:1] θ[1:2] t
# @variables x[1:NX] u[1:NU] θ[1:NΘ] t
Jx,Ju = Jacobian.([x,u])



# symbolic traces of model - how to import?
f_trace = invokelatest(f,collect(θ),collect(x),collect(u),t)
Rr_trace = invokelatest(Rr,collect(θ),collect(x),collect(u),t)
Qr_trace = invokelatest(Qr,collect(θ),collect(x),collect(u),t)


T = :SplitP





tmap(f_trace) do x
    prettify(toexpr(x))
end


# fname = tempname()*"_$T.jl"

fname = tempname()*".jl"
hdr = "#= this file was machine generated at $(now()) - DO NOT MODIFY =#\n\n"
write(fname, hdr*prod(string.(M).*"\n\n"))
    


ex = build_function(f_trace,θ,x,u,t)[2]
#todo - select inplace, [2] is unreliable
ex = crispr(ex, :ˍ₋out, :out) |> clean

# can separate this ex into args[1] - function arguments, and args[2] - body



## ------------------------------- testing ------------------------------- ##

α = 10
ω = 0.5
n = 5
v = -α/4
H0 = SymTridiagonal([4.0i^2 for i in -n:n], v*ones(2n))
w = eigvecs(H0)
x_eig = i -> SVector{NX}(kron([1;0],w[:,i]))

x0 = SVector{NX}(kron([1;0],w[:,1]))
xf = SVector{NX}(kron([1;0],w[:,2]))
u0 = 0.2
t0,tf = (0,10)

θ = SplitP(1,1)
μ = @closure t->SizedVector{1}(0.2)

α = ODE(dx_dt_ol!, xf, (t0,tf), (θ,μ), Size(x0))
x = ODE(dx_dt_ol!, x0, (t0,tf), (θ,μ), Size(x0))
u = @closure t->u_ol(θ,μ,t)

Prf = SizedMatrix{NX,NX}(I(nx(θ)))
Pr = ODE(dPr_dt!, Prf, (tf,t0), (θ,α,μ), Size(NX,NX))

x = ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))


# dx = similar(collect(x0))
# dx_dt!(dx,x0,(θ,μ(t0)),t0)




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