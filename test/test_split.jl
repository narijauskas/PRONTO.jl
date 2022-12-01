#

using PRONTO
using StaticArrays
using Symbolics
using LinearAlgebra
using MatrixEquations
using SparseArrays

using MacroTools, BenchmarkTools

# ------------------------------- model expansion ------------------------------- #

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

# ------------------------------- symbolic derivatives ------------------------------- #
using PRONTO: Jacobian
using Base: invokelatest
using PRONTO: now
using PRONTO: define, build_methods

NX = mdl.NX; NU = mdl.NU; NΘ = mdl.NΘ
# initialize variables for tracing
@variables x[1:NX] u[1:NU] θ[1:NΘ] t
ξ = vcat(x,u)
Jx,Ju = Jacobian.((x,u))
#MAYBE: x = collect(x)...


# trace symbolic
f_trace = invokelatest(mdl.f,collect(θ),t,collect(x),collect(u))
Rr_trace = invokelatest(mdl.Rr,collect(θ),t,collect(x),collect(u))
Qr_trace = invokelatest(mdl.Qr,collect(θ),t,collect(x),collect(u))

#


T = :Split
M = Expr[]
push!(M,:(
    struct $T <: PRONTO.Model{$NX,$NU,$NΘ}
        #TODO: parameter support
    end
))

def = define(f_trace,θ,ξ,t)
append!(M,build_methods(:f,T,[NX],[:θ,:ξ,:t],def))

def = PRONTO.define(Jx(f_trace),θ,t,ξ)
append!(M,PRONTO.build_methods(:Ar,T,[NX,NX],[:θ,:t,:ξ],def))

def = PRONTO.define(Ju(f_trace),θ,t,ξ)
append!(M,PRONTO.build_methods(:Br,T,[NX,NU],[:θ,:t,:ξ],def))

def = PRONTO.define(Rr_trace,θ,t,ξ)
append!(M,PRONTO.build_methods(:Rr,T,[NU,NU],[:θ,:t,:ξ],def))

def = PRONTO.define(Qr_trace,θ,t,ξ)
append!(M,PRONTO.build_methods(:Qr,T,[NX,NX],[:θ,:t,:ξ],def))

fname = tempname()*"_$T.jl"
hdr = "#= this file was machine generated at $(now()) - DO NOT MODIFY =#\n\n"
write(fname, hdr*prod(string.(M).*"\n\n"))
include(fname)


# ξ0 = SizedVector{23}([mdl.xf; 0])
ξ0 = [mdl.xf; 0]
g = @closure t->SizedVector{1}(0.2)
θ = Split()
φg = ODE(PRONTO.forced_dynamics!, collect(ξ0), (0,10), (θ,g), Buffer{Tuple{NX+NU}}(); dae = dae(θ))
Pr = SizedMatrix{NX,NX}(collect(1.0*I(NX)))
ξ = ODE(PRONTO.regulated_dynamics!, collect(ξ0), (0,10), (θ,φg,Pr), Buffer{Tuple{NX+NU}}(); dae = dae(θ))

PRONTO.regulated_dynamics!(buf,ξ,(θ,φg,Pr),t)
##






















θ = Split2()
P0 = SizedMatrix{NX,NX}(collect(1.0*I(NX)))
PRONTO.Kr(θ,0,φ,P0)
#DONE: trace & build Qr
#DONE: trace & build Rr

#DONE: benchmark matrix Kr ~ 0.6 μs
#DONE: benchmark buffered matrix Kr
#DONE: benchmark autodiff Kr ~ 8.3 μs

#TODO: register symbolic Kr(θ,t,ξ,Pr)
#TODO: make dξ_dt!


PRONTO.Kr(θ::Split,t,ξ,Pr) = PRONTO.Rr(θ,t,ξ)\(PRONTO.Br(θ,t,ξ)'Pr)



@variables K[1:NU,1:NX], x[1:NX]
def1 = define(K*x + K*x,θ,t,ξ)
build_methods(:Kr1,T,[NU,NX],[:θ,:t,:ξ],def1)
def2 = define(collect(K*x),θ,t,ξ)















function Kr1(θ,t,ξ,Pr)
    PRONTO.Rr(θ,t,ξ)\PRONTO.Br(θ,t,ξ)'Pr
end

function Kr2(θ,t,ξ,Pr)
    buf = SizedMatrix{NU,NX,Float64}(undef)
    mul!(buf,PRONTO.Br(θ,t,ξ)',Symmetric(Pr))
    Diagonal(PRONTO.Rr(θ,t,ξ))\buf
end

@variables P[1:NX,1:NX]
Pr = Symmetric(P)
args = [:θ,:t,:ξ, :Pr]
def = PRONTO.define(Rr_trace\Ju(f_trace)'*Pr,θ,t,ξ,Pr);
append!(M,PRONTO.build_methods(:Kr,T,[NX,NU],args,def));


##




















# # assemble function
# f_fn = eval(build_function(f_tr,θ,t,ξ)[1])
# f_ip = eval(build_function(f_tr,θ,t,ξ)[2])


fx_trace = Jx(f_tr)
fu_trace = Ju(f_tr)


α = rand(22)
μ = rand(1)
φ = vcat(α,μ)
out = zeros(22)

@benchmark mdl.f([],0,α,μ) # 3010 ns
@benchmark f_fn([],0,φ) # 407 ns
@benchmark f_ip(out,[],0,φ) # 333 ns



## ------------------------------- symbolic construction of odes ------------------------------- ##

Kr(B,P,R) = R\(B'P)
# Kr(θ,t,φ) = Kr(B(θ,t,φ), ...)
function dPr_dt!(dPr,Pr,(A,B,Q,R),t)
    dPr .=  -A'P - P*A + P'B*(R\B'P) - Q
end

function riccati(A,B,P,Q,R)
    # uses expanded K'R*K
    -A'P - P*A + P'B*(R\B'P) - Q
end

dPr = unroll(riccati(fx_trace, fu_trace,Pr,Qr,Rr))

# Kr??
# Pr??

fx_tr = Jx(f_tr) #Ar
fu_tr = Ju(f_tr) #Br


function dξ_dt()
    f(x,u)
    μ - Kr(θ,t,φ,Pr)*(x-α) - u
end
# trace, build:
dξ_dt!(out, θ, t, ξ, φ, Pr)


# build a function for each Br[i] -> generator of those functions
build_function(collect(fu_tr[1,1]), θ, t, ξ)
Br_fns = eval.(build_function(fu_tr[i,j], θ, t, ξ) for i in 1:NX, j in 1:NU)
Br_fn = eval(build_function(fu_tr, θ, t, ξ)[1])
Br_fn! = eval(build_function(fu_tr, θ, t, ξ)[2])
[Br([],0,φ) for Br in Br_fns] # 3 μs
function fn1(Br_fns,φ)
    [Br([],0,φ) for Br in Br_fns]
end
@time fn1(Br_fns,φ) # 3 μs

@benchmark Br_fn([],0,φ) # 600 ns
@benchmark Br_fn!(out,[],0,φ) # 280 ns
@benchmark Br_fn2([],0,φ) # 354 ns


Br_fn2(θ,t,ξ) = let Br! = Br_fn!
    out = SizedMatrix{22,1}(zeros(22,1))
    Br!(out,θ,t,ξ)
    return out
end


Ar_1 = eval(build_function(Jx(f_trace), θ, t, ξ)[1])
@benchmark Ar_1([],0,φ) # 1.2 ms

Ar_2 = let Ar! = eval(build_function(Jx(f_trace), θ, t, ξ)[2])
    function _Ar(θ,t,ξ)
        out = SizedMatrix{22,22,Float64}(undef)
        Ar!(out,θ,t,ξ)
        return out
    end
end

@benchmark Ar_2([],0,φ) # 535.8 ns




using Base.Threads: @spawn

tsk = map(1:400000) do i
    @spawn Ar_2([],0, i .+ rand(23))
end









abstract type AbstractModel{NX,NU,NΘ} <: FieldVector{NΘ,Float64} end

struct Yeet <: AbstractModel{22,1,0}
end



@build Br{$NX,$NU}(θ,t,ξ) = Ju(f_trace)












function separate(ex)
    @capture(ex, name_{dims__}(args__) = body_)
    return (name,dims,args)
end

(name,dims,args) = separate(ex)


PRONTO.@build Br{$NX,$NU}(θ,t,ξ) 

ex = :(Br{$NX,$NU}(θ,t,ξ) = Ju(f_trace))
separate(ex)


hdr = ex.args[1]
@capture(ex, name_{dims__}(args__) = body_)


# expects Ju(f_trace) will evaluate with locally defined (θ,t,ξ)
@build Br{NX,NU}(θ,t,ξ) = Ju(f_trace)

build_function(fu_tr, θ, t, ξ)[2] # |> rename Br!(out,θ,t,ξ)
# |> rename fxn!(out,args...)

# define
function Br(args...)
    out = SizedMatrix{22,1}(zeros(22,1))
    Br!(out,θ,t,ξ)
    return out
end





using StaticArrays: sacollect
sacollect(SMatrix{NX,NU}, Br_fns[i,j]([],0,φ) for i in 1:NX, j in 1:NU)
fn2(Br_fns,φ) = sacollect(SMatrix{22,1}, Br_fns[i,j]([],0,φ) for i in 1:22, j in 1:1)
fn3(Br_fns,φ) = sacollect(SMatrix{22,1}, Br_fns[i,j]([],0,φ)::Float64 for i in 1:22, j in 1:1)
fn4(Br_fns,φ) = sacollect(SMatrix{22,1,Float64}, Br([],0,φ) for Br in Br_fns)
fn5(Br_fns,φ) = SA_F64[Br_fns[i,j]([],0,φ) for i in 1:22, j in 1:1]

function fn1!(out,Br_fns,φ)
    for i in eachindex(out)
        out[i] = Br_fns[i]([],0,φ)
    end
end

function fn2!(out,Br_fns,φ)
    for (i,Br) in enumerate(Br_fns)
        setindex!(out, Br([],0,φ), i)
    end
end






using FunctionWrappers: FunctionWrapper
FunctionWrapper{Float64, Tuple{Vector,Number,Vector{Float64}}}((θ,t,ξ)->Br_fns[1,1](θ,t,ξ))
Br_w = FunctionWrapper{Float64, Tuple{Vector,Number,Vector{Float64}}}.((θ,t,ξ)->Br(θ,t,ξ) for Br in Br_fns)


@benchmark fn2(Br_fns,φ) # 4.1 μs
@benchmark fn2(Br_w,φ) # 1.6 μs
@benchmark fn3(Br_fns,φ) # 3.1 μs
@benchmark fn4(Br_fns,φ) # 4.6 μs
@benchmark fn5(Br_fns,φ) # 3.7 μs


out = zeros(22,1)
@benchmark fn1!(out,Br_fns,φ)
@benchmark fn1!(out,Br_w,φ)
@benchmark fn2!($out,$Br_fns,$φ)



out[i] = $(Br_fns[i])([],0,$φ)
@benchmark begin # 0.7 μs
    setindex!($out, $(Br_fns[1])([],0,$φ), 1)
    setindex!($out, $(Br_fns[2])([],0,$φ), 2)
    setindex!($out, $(Br_fns[3])([],0,$φ), 3)
    setindex!($out, $(Br_fns[4])([],0,$φ), 4)
    setindex!($out, $(Br_fns[5])([],0,$φ), 5)
    setindex!($out, $(Br_fns[6])([],0,$φ), 6)
    setindex!($out, $(Br_fns[7])([],0,$φ), 7)
    setindex!($out, $(Br_fns[8])([],0,$φ), 8)
    setindex!($out, $(Br_fns[9])([],0,$φ), 9)
    setindex!($out, $(Br_fns[10])([],0,$φ), 10)
    setindex!($out, $(Br_fns[11])([],0,$φ), 11)
    setindex!($out, $(Br_fns[12])([],0,$φ), 12)
    setindex!($out, $(Br_fns[13])([],0,$φ), 13)
    setindex!($out, $(Br_fns[14])([],0,$φ), 14)
    setindex!($out, $(Br_fns[15])([],0,$φ), 15)
    setindex!($out, $(Br_fns[16])([],0,$φ), 16)
    setindex!($out, $(Br_fns[17])([],0,$φ), 17)
    setindex!($out, $(Br_fns[18])([],0,$φ), 18)
    setindex!($out, $(Br_fns[19])([],0,$φ), 19)
    setindex!($out, $(Br_fns[20])([],0,$φ), 20)
    setindex!($out, $(Br_fns[21])([],0,$φ), 21)
    setindex!($out, $(Br_fns[22])([],0,$φ), 22)
end

Br_w[1]([],0,φ)

function fn4(Br_fns,φ)
    temp = sacollect(SMatrix{22,1}, Br_fns[i,j]([],0,φ) for i in 1:22, j in 1:1)
    sum(temp'temp)
end


@variables P[1:NX,1:NX] Q[1:NX,1:NX] R[1:NU,1:NU]
Ar = sparse(fx_tr)
Br = sparse(fu_tr)
Pr = Symmetric(P)
Qr = collect(Diagonal(collect(Q)))
Rr = collect(Diagonal(collect(R)))
Kr(Br,Pr,Rr)
@time dPr_trace = riccati(Ar,Br,Pr,Qr,Rr);
dPr_11 = dPr_trace[1,1]
dPr_fn = build_function(dPr_11,θ,t,ξ,Pr,Qr,Rr)
R0 = [1.0;;]
Q0 = diagm(ones(NX))
P0 = Symmetric(rand(NX,NX))
dPr_fn([],0,φ,P0,Q0,R0) # 636 ns

@variables A[1:NX,1:NX] B[1:NX,1:NU]
A = PRONTO.sparse_mask(A,Ar)
B = PRONTO.sparse_mask(B,Br)
@time dPr_trace = riccati(A,B,Pr,Qr,Rr);
dPr_11 = dPr_trace[1,1]
dPr_fn = eval(build_function(dPr_trace,collect(A),collect(B),P,Q,R)[1])

A0 = PRONTO.sparse_mask(ones(NX,NX),Ar)
B0 = PRONTO.sparse_mask(ones(NX,NU),Br)

R0 = [1.0;;]
Q0 = diagm(ones(NX))
P0 = Symmetric(rand(NX,NX))

@benchmark riccati(A0,B0,P0,Q0,R0) # approx 167ns
dPr_fn(A0,B0,P0,Q0,R0)




# @variables Q[1:NX,1:NX]
# QQ = collect(Q).*Int.(I(NX)) # extract diagonal
#NOTE: inv(QQ'*QQ)*QQ' works better with symbolics than inv(QQ)



# goals:
# 1. construct symbolic & manual dPr_dt!, compare
# 2. triangular dPr - can we do this via trace?
# alt: create unrolling function mapping symbolic ricatti matrix to vector of actual values
# unroll(M) -> V
# roll(V) -> symmetric(M)



















#= ----------------------------------- symbolic derivatives ----------------------------------- =#

fn = build_function(Ju(Ju(f_tr)),θ,t,ξ)
open(fname, "w+") do io
    code_native(io,fn,[Vector{Float64},Vector{Any},Float64,Vector{Float64}])
end

fnp = eval(build_function(Ju(Ju(f_tr)), θ,t,ξ, parallel=Symbolics.MultithreadedForm())[2])
fn = eval(build_function(Ju(Ju(f_tr)), θ,t,ξ)[2])
out = MVector(zeros(22)...)
φ = SVector(rand(23)...)
fn(out,[],0,φ); out
@benchmark fn(out,[],0,φ)
@benchmark fnp(out,[],0,φ)


@profview test_fn(fn,out,φ,10000)

function test_fn(fn,out,φ,n)
    j = 0
    for i in 1:n
        fn(out,[],0,φ)
        j += i
    end
    return out
end









Jx,Ju,f,l,p,Q,R = PRONTO.model(T,ex)
##
riccati(A,K,P,Q,R) = -A'P - P*A + Symmetric(K'R*K) - Q

A = Jx(ff)
B = Ju(ff)
@variables Pr[1:22,1:22]
Kr = collect(Rr\collect(B'*Pr))
riccati(A,Kr,Pr,Qr,Rr)
##
Ar .* map(Jx(f)) do ex
    iszero(ex) ? 0 : 1
end |> collect |> sparse

sparse_mask(Ar, Jx(f))

sparse(collect(x))
x |> collect |> sparse

M = Split()
θ = Float64[]
# x0 = [2π/3;0]
u0 = [0.0]
ξ0 = [x0;u0]
# ξf = [0;0;0]
t0 = 0.0; tf = 10.0

##
φg = @closure t->ξ0
φ = guess_φ(M,θ,ξf,t0,tf,φg)

ug = @closure t -> u0
φg = PRONTO.guess_ol(M,θ,t0,tf,x0,ug)
##
pronto(M,θ,t0,tf,x0,u0,φg)
# @time ξ = pronto(M,θ,t0,tf,x0,u0,φg; tol = 1e-8, maxiters=100)




function fn2(out,A,P,K,R,Q)
    out .-= A'*P
    out .-= P*A
    out .-= Q
    # out .-= K'*R*K
    return out
end


function fn2(A,P)
    .- A'*P .- P*A
end

function fn3(A,P)
    .- A'*P .- P*A
end

##
#variables
@variables P[1:NX,1:NX]
@variables R[1:NU,1:NU]
@variables Q[1:NX,1:NX]
@variables A[1:NX,1:NX]
@variables B[1:NX,1:NU]

Ar = PRONTO.sparse_mask(A, Jx(f))
Br = PRONTO.sparse_mask(B, Ju(f))
Kr = PRONTO.regulator(B,P,R)
PRONTO.riccati(Ar,Kr,P,Q,R)
-Ar'*Pr - Pr*Ar + Kr'*R*Kr - Q