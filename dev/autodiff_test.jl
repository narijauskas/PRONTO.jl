using PRONTO
using ForwardDiff
using DifferentialEquations

H0 = [0 0 1 0;
      0 0 0 -1;
     -1 0 0 0;
      0 1 0 0]

H1 = [0 -1 0 0;
      1 0 0 0;
      0 0 0 -1;
      0 0 1 0]

 f(x,u,t) =(H0 + u*H1)*x;

fx(x,u,t) = H0 + u*H1;
fu(x,u,t) = H1*x;

T = 15
tspan = (0.0,T)

dmu(mu,_,t) = 0*t

mu = solve(ODEProblem(dmu,0,tspan))

dal(al,_,t) = f(al,mu(t))

al0 = [1;0;0;0]

al = solve(ODEProblem(dal,al0,tspan))

function inprod(x)
    a = x[1:2]
    b = x[3:4]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a' a*a'+b*b']
    return P
end

Pr_terminal(al) = I(4) - inprod(al)

Ar(al,mu,t) = fx(al,mu,t)
Br(al,mu,t) = fu(al,mu,t)
Qr(al,mu,t) = I(4) - inprod(al)
Rr(al,mu,t) = 1
    
Ar(t) = Ar(al(t),mu(t),t)
Br(t) = Br(al(t),mu(t),t)
Qr(t) = Qr(al(t),mu(t),t)

Kr(Pr,Rr,Br,t)= Rr\(Br'*Pr)
    
Kr(t) = Kr(Pr(t),Rr(t),Br(t),t)

dPr(Pr,_,t) = Ar(T-t)'*Pr + Pr*Ar(T-t) - Kr(Pr,Rr(T-t),Br(T-t),T-t)'*Rr(T-t)*Kr(Pr,Rr(T-t),Br(T-t),T-t) + Qr(T-t)
Pr(t) = Pr_back(T-t)
    
u(x,Kr,al,mu,t) = mu - Kr*(x - al)

u(t) = u(x(t),Kr(t),al(t),mu(t),t)
    
dx(x,_,t) = f(x,u(x,Kr(Pr(t),Rr(t),Br(t),t),al(t),mu(t),t))

x0 = [0;1;0;0]

Pr_back = solve(ODEProblem(dPr,Pr_terminal(al(T)),tspan))

x = solve(ODEProblem(dx,x0,tspan))