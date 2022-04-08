import Pkg; Pkg.activate(".")
using BenchmarkTools
using Symbolics
using PRONTO
using PRONTO: jacobian
using DataInterpolations
using MatrixEquations
using DifferentialEquations
using LinearAlgebra

##
@variables x[1:4] u[1:1]
# dynamics
H0 = [0 0 1 0;
      0 0 0 -1;
     -1 0 0 0;
      0 1 0 0]

H1 = [0 -1 0 0;
      1 0 0 0;
      0 0 0 -1;
      0 0 1 0]

f(x,u) = collect((H0 + u[1]*H1)*x)


fx = jacobian(x, f, x, u)
fu = jacobian(u, f, x, u)

fxx = jacobian(x, fx, x, u)
fxu = jacobian(u, fx, x, u)
fuu = jacobian(u, fu, x, u)

fx(x,u)
##

T = 15
t = 0:0.01:T 
X = LinearInterpolation(zeros(4, length(t)), t) 
U = LinearInterpolation(zeros(length(t)), t)

dal(al,_,t) = f(al,U(t))
al0 = [1;0;0;0]
al = solve(ODEProblem(dal,al0,(0,T)))

X = LinearInterpolation(al.u, al.t)

##

Ar = t -> fx(X(t), U(t))
Br = t -> fu(X(t), U(t))

function inprod(x)
    a = x[1:2]
    b = x[3:4]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a' a*a'+b*b']
    return P
end

# lqr stage cost
Qr(t) = I(4) - inprod(X(t))
Rr(t) = 1
Pr_terminal = I(4) - inprod(X(T))

function riccati!(dPr, Pr, (Ar,Br,Qr,Rr), t)
    Kr = inv(Rr(t))*Br(t)'*Pr
    dPr .= -Ar(t)'Pr - Pr*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
end

Prob = ODEProblem(riccati!, Pr_terminal, (T, 0.0), (Ar,Br,Qr,Rr))
Pr = solve(Prob)  

Kr = t->inv(Rr(t))*Br(t)'*Pr(t)

##