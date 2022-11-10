
using PRONTO
using FastClosures
using StaticArrays
using LinearAlgebra
using MatrixEquations


NX = 22
NU = 1
NΘ = 0

struct Split <: PRONTO.Model{NX,NU,NΘ}
end

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
    global x0 = [nu1;0*nu1]
    global xf = [nu2;0*nu2]

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



##

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
##
@time ξ = pronto(M,θ,t0,tf,x0,u0,φ; tol = 1e-8, maxiters=100)
