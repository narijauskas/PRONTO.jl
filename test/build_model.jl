
## --------------------------- autodiff --------------------------- ##

model = merge(model, (
    fx = jacobian(x, model.f, x, u),
    fu = jacobian(u, model.f, x, u),
))
model = merge(model, (
    fxx = jacobian(x, model.fx, x, u),
    fxu = jacobian(u, model.fx, x, u),
    fuu = jacobian(u, model.fu, x, u),
))
model = merge(model, (
    lx = jacobian(x, model.l, x, u),
    lu = jacobian(u, model.l, x, u),
))
model = merge(model, (
    lxx = jacobian(x, model.lx, x, u),
    lxu = jacobian(u, model.lx, x, u),
    luu = jacobian(u, model.lu, x, u),
))

## --------------------------- terminal cost --------------------------- ##

xt = model.x_eq
ut = model.u_eq

A = model.fx(xt, ut)
B = model.fu(xt, ut)
Q = model.lxx(xt, ut)
R = model.luu(xt, ut)
S = model.lxu(xt, ut)
Po,_ = arec(A, B, R, Q ,S)
p = x -> 1/2*collect(x)'*Po*collect(x)

model = merge(model, (p=p,))
model = merge(model, (px = jacobian(x, model.p, x),))
model = merge(model, (pxx = jacobian(x, model.px, x),))


