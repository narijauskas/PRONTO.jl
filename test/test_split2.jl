
using StaticArrays, LinearAlgebra, Symbolics
using SparseArrays, MatrixEquations
using MacroTools, BenchmarkTools
##
using PRONTO
##
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


## ------------------------------- model def ------------------------------- ##
# for now:
NX = 22; NU = 1

struct SplitP <: Model{22,1,2}
    kr::Float64
    kq::Float64
end


function f(θ,x,u,t)
    α = 10
    ω = 0.5
    n = 5
    v = -α/4
    H0 = SymTridiagonal([4.0i^2 for i in -n:n], v*ones(2n))
    H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    H2 = v*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
    return mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x
end


# for now, this will work:
Rr(θ,x,u,t) = θ[1]*I(NU)
Qr(θ,x,u,t) = θ[2]*I(NX)










## ------------------------------- symbolic derivatives ------------------------------- ##
using PRONTO: Jacobian
using Base: invokelatest
using PRONTO: now, crispr, clean
using PRONTO: define, build_methods



# iinfo("initializing symbolics\n")
# create symbolic variables & operators
@variables x[1:22] u[1:1] θ[1:2] t
# @variables x[1:NX] u[1:NU] θ[1:NΘ] t
Jx,Ju = Jacobian.([x,u])




# symbolic traces of model - how to import?
f_trace = invokelatest(f,collect(θ),collect(x),collect(u),t)
Rr_trace = invokelatest(Rr,collect(θ),collect(x),collect(u),t)
Qr_trace = invokelatest(Qr,collect(θ),collect(x),collect(u),t)




ex = build_function(f_trace,θ,x,u,t)[2]
#todo - select inplace, [2] is unreliable
ex = crispr(ex, :ˍ₋out, :out) |> clean

# can separate this ex into args[1] - function arguments, and args[2] - body

## ------------------------------- symbolic derivatives ------------------------------- ##


@capture(ex, :(function (args__) body_ end))


M = Expr[]









α = 10
ω = 0.5
n = 5
v = -α/4
H0 = SymTridiagonal([4.0i^2 for i in -n:n], v*ones(2n))
H1 = v*im*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))
H2 = v*Tridiagonal(ones(2n), zeros(2n+1), -ones(2n))


f(θ,x,u,t) = mprod(-im*ω*(H0 + sin(u[1])*H1 + (1-cos(u[1]))*H2) )*x




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