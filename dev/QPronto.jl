import Pkg; Pkg.activate(".")
using BenchmarkTools
using Symbolics
using Symbolics: derivative
# using PRONTO
# using PRONTO: jacobian
using DataInterpolations
using MatrixEquations
using DifferentialEquations
using LinearAlgebra

##
@variables x[1:4] u[1:1]

H0 = [0 0 1 0;
      0 0 0 -1;
     -1 0 0 0;
      0 1 0 0]

H1 = [0 -1 0 0;
      1 0 0 0;
      0 0 0 -1;
      0 0 1 0]

f(x,u) = collect((H0 + u[1]*H1)*x)

function jacobian(dx, f, args...; inplace = false)
    f_sym = f(args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(1:length(f_sym)) do j
            derivative(f_sym[j], dx[i])
        end
    end...; dims = ndims(f_sym)+1)

    return eval(build_function(fx_sym, args...)[inplace ? 2 : 1])
end


fx = jacobian(x, f, x, u)
fu = jacobian(u, f, x, u)

fxx = jacobian(x, fx, x, u)
fxu = jacobian(u, fx, x, u)
fuu = jacobian(u, fu, x, u)

fx(x,u)

T = 15
t = 0:0.01:T 
# X = LinearInterpolation(zeros(4, length(t)), t) 
mu = LinearInterpolation(zeros(length(t)), t)

dal(al,_,t) = f(al,mu(t))
al0 = [1;0;0;0]
al = solve(ODEProblem(dal,al0,(0,T)))

# X = LinearInterpolation(al.u, al.t)

Ar = t -> fx(al(t), mu(t))
Br = t -> fu(al(t), mu(t))

function inprod(x)
    a = x[1:2]
    b = x[3:4]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a' a*a'+b*b']
    return P
end


Qr(t) = I(4) - inprod(al(t))
Rr(t) = 1
Pr_terminal = I(4) - inprod(al(T))

function riccati!(dPr, Pr, (Ar,Br,Qr,Rr), t)
    Kr = inv(Rr(t))*Br(t)'*Pr
    dPr .= -Ar(t)'Pr - Pr*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
end

Prob = ODEProblem(riccati!, Pr_terminal, (T, 0.0), (Ar,Br,Qr,Rr))
Pr = solve(Prob)  

Kr = t->inv(Rr(t))*Br(t)'*Pr(t)

##

function dynamics!(dX, X, (al,mu,Kr), t)
    U = mu(t) - (Kr(t)*(X - al(t)))[1]
    dX .= f(X,U)
end

X0 = [0;1;0;0]

Prob1 = ODEProblem(dynamics!, X0, (0.0, T), (al,mu,Kr))
X = solve(Prob1)
U(t) = mu(t) - (Kr(t)*(X(t) - al(t)))[1]
##

l(x,u) = 0.01/2*u[1]^2

lx = jacobian(x, l, x, u)
lu = jacobian(u, l, x, u)

lxx = jacobian(x, lx, x, u)
lxu = jacobian(u, lx, x, u)
luu = jacobian(u, lu, x, u)

##

A = t -> fx(X(t), U(t))
B = t -> fu(X(t), U(t))