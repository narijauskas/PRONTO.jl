
NX = 22; NU = 1; NΘ = 0
struct Split <: Model{NX,NU,NΘ}
end

# 1. define a struct however you like (@kwdef should work...)
# 2. the struct must be a subtype of Model{NX,NU,NΘ}, where those encode dimensions
# 3. fields = parameters, and can be accessed by, eg. θ.k or θ[1]

NX = 22; NU = 1; NΘ = 4
struct SplitP <: Model{NX,NU,NΘ}
    kr::Float64
    kq::Float64
    α::Float64 # 10
    ω::Float64 # 0.5
end



@define SplitP begin
    

    f(θ,x,u,t) = mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x


    # how to trace these? reverse lookup on type?
    # Rr(θ,x,u,t) = θ.kr*I(nu(θ))
    # Qr(θ,x,u,t) = θ.kq*I(nx(θ))

    # for now, this will work:
    Rr(θ,x,u,t) = θ[1]*I(nu(θ))
    Qr(θ,x,u,t) = θ[2]*I(nx(θ))



end

import LinearAlgebra: Tridiagonal, SymTridiagonal
SymTridiagonal(dv,ev) = SymTridiagonal(promote(dv,ev)...)
Tridiagonal(dl,d,du) = Tridiagonal(promote(dl,d,du)...)

@variables θ[1:NΘ] x[1:NX] u[1:NU]
α = θ[3]
ω = θ[4]
n = 5 # length of side
v = -α/4
H0 = SymTridiagonal([4.0i^2 for i in -n:n], v*ones(2n))
H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
H2 = v*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))




function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end








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