using Test
using PRONTO
using PRONTO: jacobian
using Symbolics

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
fx = fx_auto = jacobian(x, f, x, u)
fu = fu_auto = jacobian(u, f, x, u)

fx_manual(x,u) = [
    0 -u[1] 1 0;
    u[1] 0 0 -1;
    -1 0 0 -u[1];
    0 1 u[1] 0;
]

fu_manual(x,u) = [-x[2]; x[1]; -x[4]; x[3];;]

@test isequal(fx_manual(x,u), fx_auto(x,u))
@test isequal(fu_manual(x,u), fu_auto(x,u))