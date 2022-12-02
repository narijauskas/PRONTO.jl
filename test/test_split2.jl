
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




# PRONTO.define(f,Rr,Qr)

generate_methods(SplitP,f,Rr,Qr)


## ------------------------------- symbolic derivatives ------------------------------- ##
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
M = [
    build_inplace(T, :f!, f_trace),
    build(Size(NX), T, :f, f_trace),
    build(Size(NX,NX), T, :Ar, Jx(f_trace)),
    build(Size(NX,NU), T, :Br, Ju(f_trace)),
    build(Size(NX,NX), T, :Qr, Qr_trace),
    build(Size(NU,NU), T, :Rr, Rr_trace)
];




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
xg = i -> SVector{NX}(kron([1;0],w[:,i]))

x0 = SVector{NX}(kron([1;0],w[:,1]))
xf = SVector{NX}(kron([1;0],w[:,2]))
u0 = 0.2
t0,tf = (0,10)

θ = SplitP(1,1)
μ = @closure t->SizedVector{1}(0.2)

α = ODE(dx_dt!, xf, (t0,tf), (θ,u_ol(θ,μ,t)), Size(x0))
x = ODE(dx_dt!, x0, (t0,tf), (θ,u_ol(θ,μ,t)), Size(x0))
u = @closure t->u_ol(θ,μ,t)

Prf = SizedMatrix{NX,NX}(I(nx(θ)))
Pr = ODE(dPr_dt!, Prf, (tf,t0), (θ,α,μ), Size(NX,NX))

x = ODE(dx_dt!, x0, (t0,tf), (θ,α,μ,Pr), Size(x0))


# dx = similar(collect(x0))
# dx_dt!(dx,x0,(θ,μ(t0)),t0)




# this is insanely cool:
μ = @closure t->SizedVector{1}(0.1)

for x0 in xg.(1:11)
    show(ODE(dx_dt_ol!, x0, (t0,tf), (θ,μ), Size(x0)))
end





























@capture(ex, :(function (args__) body_ end))


M = Expr[]

tmap(f_trace) do x
    string(prettify(toexpr(x)))
end

trace_symbolic(f) = invokelatest(f, collect.(args)...)

# need to know:
# name, size, args/T
build_pretty(name, T, syms) = build_expr(name, T, syms; postprocess = prettify)

function build_expr(name, T, syms; postprocess = identity)
    # parallel map each symbolic to expr
    defs = tmap(enumerate(syms)) do (i,x)
        :(out[$i] = $(postprocess(toexpr(x))))
    end

    return quote
        function PRONTO.$name(θ::$T,x,u,t)
            out = SizedMatrix{$NX,$NX,Float64}(undef)
            @inbounds begin
                $(defs...)
            end
            return out
        end
    end |> clean
end

M1 = build_expr(:f1, :SplitP, f_trace)
M2 = build_pretty(:f1, :SplitP, f_trace)





ex = quote
    function Foo(b,a,r)
        out = SizedMatrix{$NX,$NX,Float64}(undef)
        @inbounds begin $(M...) end
        return out
    end
end

clean(ex)

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