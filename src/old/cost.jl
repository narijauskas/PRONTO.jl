using QuadGK

wnorm(vec, mat) = 1/2 * vec'*mat*vec

## ------------------------------ typical cost functional ------------------------------ ## 
 
function build_LQ_cost(ξd, Q, R, P1, T)
    # err = (ξ, t) -> ξ(t) - ξd(t)
    l = (x, u, t) -> wnorm( (x(t)-ξd.x(t)), Q) + wnorm( (u(t)-ξd.u(t)), R)
    m = xT -> wnorm( (xT-ξd.x(T)), P1) 
    return l, m
end

# function h(ξd, Q, R, P1, T)
#     return ξ -> LQ_cost(ξ, ξd, Q, R, P1, T)
# end

# function LQ_cost_dt(ξ, ξd, Q, R, t)
#     # add time derivative of LQ_cost
#     err = ξ - ξd
#     return wnorm(err.x(t), Q(t)) + wnorm(err.u(t), R(t))
# end 

# function DLQ_cost(ζ, ξ, ξd, Q, R, P1, T) 
#     # add total derivative of LQ_cost
#     err = ξ - ξd
#     l = t -> err.x(t)' * Q * ζ.x + err.u(t)' * R * ζ.u
#     L, _ = quadgk(l, 0, T)
#     L += err.x(T)' * P1 * ζ.x(T) # terminal cost
#     return L
# end

## ------------------------------ Generic cost functions ------------------------------ ## 
 
# should have form: 
# incremental cost: l(x, u, t) = build_l(p, t) where x(t), u(t) evaluated externally
# terminal cost: m(x(T)) 
# h = ∫ l(ξ, τ) dτ + m(x(T)) 

# l(x, u, t) = l(Trajectory(x,u), t)

function loss_grads1(l, m, ξ, T)
    lx = t -> gradient(x -> l(x, ξ.u, t), ξ.x(t)) # time evaled internally# return fn lx(t)
    lu = t -> gradient(u -> l(ξ.x, u, t), ξ.u(t)) # return fn lu(t)
    a = t -> lx(t)' # return lx(ξ)'
    b = t -> lu(t)'

    # ḣ = l # time derivative of full cost fn is just l
    mx = gradient(x -> m(x), ξ.x(T))' # constant for given ξ
    r1 = mx'
    return a, b, r1 # functions of time
end

function loss_grads2(l, m, ξ, T)
    lxx = t -> hessian(x -> l(x, ξ.u(t), t), ξ.x(t)) # return fn lxx(t)
    luu = t -> hessian(u -> l(ξ.x(t), u, t), ξ.u(t)) # return fn lxx(t)
    lxu = t -> jacobian(u ->  gradient(x -> l(x, u(t), t), ξ.x(t)), ξ.u(t)) # ??? # return fn lxu(t)
    mxx = hessian(x -> m(x), ξ.x(T))'
    P1 = mxx #TODO: Check pos def?
    return lxx, lxu, luu, P1
end

build_h(l, m, ξ, T) = quadgk(t -> l(ξ.x(t), ξ.u(t), t), 0, T) + m(ξ(T)) # return h(ξ)

function tot_grads(ζ, a, b, Q₀, R₀, S₀)
    Dh = t -> a(t)' * ζ.x(t) + b(t)' * ζ.u(t)
    D2g = t -> wnorm(ζ.x(t), Q₀(t)) + 2ζ.z(t)'*S₀(t)*ζ.x(t) + wnorm(ζ.x(t), R₀(t))
    return Dh, D2g
end

build_Dh(a, b, r1) = ζ -> quadgk(t -> a(t)*ζ.x(t) + b(t)*ζ.u(t), 0, T) + r1' * ζ.x(T)

# example of how to get all cost quantities
# loss_grads1_ml = (ξ, T) -> loss_grads1(l, m, ξ, T) # for l, m in scope, constant
# loss_grads2_ml = (ξ, T) -> loss_grads2(l, m, ξ, T) # for l, m in scope, constant
# a, b, ḣ, r1 = loss_grads1_ml(l, m)
# lxx, lxu, luu, P1 = loss_grads2ml(l, m, ξ, T)
# h(ξ, T) =  build_h(l, m, ξ, T)
# Dh, D2g = tot_grads(ζ, a, b, Q₀, R₀, S₀)

# begin
#     z = ζ.x; v = ζ.u; x = ξ.x; u = ξ.u
#     gradient(x->h(?), z) + gradient(u->h(?), v)
# end