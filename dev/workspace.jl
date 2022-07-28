x = Interpolant(t->zeros(NX), ts)
u = Interpolant(t->zeros(NU), ts)

α = Interpolant(t->zeros(NX), ts)
μ = Interpolant(t->zeros(NU), ts)

Kr = PRONTO.regulator(NX,NU,T,α,μ,model.fx!,model.fu!,model.iRr,model.Rr,model.Qr);

_x = PRONTO.projection_x(NX,T,α,μ,Kr,model.f,model.x0);
PRONTO.update!(x, _x)
_u = PRONTO.projection_u(NX,NU,α,μ,Kr,x);
PRONTO.update!(u, _u)

Ko = PRONTO.optimizer(NX,NU,T,x,u,α,model.fx!,model.fu!,model.lxx!,model.luu!,model.lxu!,model.pxx!);

vo = PRONTO.costate_dynamics(NX,NU,T,x,u,α,Ko,model.fx!,model.fu!,model.lx!,model.lu!,model.luu!,model.px!);
