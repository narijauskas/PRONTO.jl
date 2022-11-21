
using PRONTO
using FastClosures
using StaticArrays
using LinearAlgebra
using MatrixEquations



function mprod(x)
    Re = I(2)  
    Im = [0 -1;
           1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

function inprod(x)
    NX = length(x)
    a = x[1:Int(NX/2)]
    b = x[(Int(NX/2)+1):(2*Int(NX/2))]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a' a*a'+b*b']
    return P
end


##
let
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
    x0 = [nu1;0*nu1]
    xf = [nu2;0*nu2]

    f = (θ,t,x,u) -> collect(mprod(-1im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2))*x)
    
    Ql = zeros(2*n,2*n)
    Rl = I
    l = (θ,t,x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
    
    Rr = (θ,t,x,u) -> diagm(ones(1))
    Qr = (θ,t,x,u) -> diagm(ones(2*n))
    
    p = (θ,t,x,u) -> begin
        P = I(2*n) - inprod(xf)
        1/2*collect(x)'*P*collect(x)
    end


    @derive Split
end
# T = :Split
# ex = quote

@model Split begin
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