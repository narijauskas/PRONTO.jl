
using PRONTO
using FastClosures
using StaticArrays
using Symbolics
using LinearAlgebra
using MatrixEquations


using MacroTools, BenchmarkTools

#= ----------------------------------- model expansion ----------------------------------- =#

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

#= ----------------------------------- symbolic derivatives ----------------------------------- =#
using PRONTO: Jacobian
using Base: invokelatest

NX = mdl.NX; NU = mdl.NU; NΘ = mdl.NΘ
# initialize variables for tracing
@variables x[1:NX] u[1:NU] θ[1:NΘ] t
ξ = vcat(x,u)
Jx,Ju = Jacobian.((x,u))
#MAYBE: x = collect(x)...


# trace symbolic
f_tr = invokelatest(mdl.f,collect(θ),t,collect(x),collect(u))
# assemble function
f_fn = eval(build_function(f_tr,θ,t,ξ)[1])
f_ip = eval(build_function(f_tr,θ,t,ξ)[2])


fx_tr = Jx(f_tr)
fu_tr = Ju(f_tr)

α = rand(22)
μ = rand(1)
φ = vcat(α,μ)
out = zeros(22)

@benchmark mdl.f([],0,α,μ) # 3010 ns
@benchmark f_fn([],0,φ) # 407 ns
@benchmark f_ip(out,[],0,φ) # 333 ns



#= ----------------------------------- symbolic construction of odes ----------------------------------- =#




# Kr??
# Pr??

fx_tr = Jx(f_tr) #Ar
fu_tr = Ju(f_tr) #Br

@variables P[1:NX,1:NX]
Pr = Symmetric(P)

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