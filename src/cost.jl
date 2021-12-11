using QuadGK

wnorm(vec, mat) = 1/2 * vec'*mat*vec

## ------------------------------ typical cost functional ------------------------------ ## 
 
function build_LQ_cost(ξd, Q, R, P1, T)
    err = (ξ, t) -> ξ(t) - ξd(t)
    l = (ξ, t) -> wnorm(err(ξ, t).x, Q(t)) + wnorm(err(ξ, t).u, R(t))
    m = ξ -> wnorm(err(ξ, T).x, P1) 
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
# incremental cost: l(ξ, t) = build_l(ξ, p, t)
# terminal cost: m(ξ) 
# h = ∫ l(ξ, τ) dτ + m(ξ(T)) 

function loss_grads1(l, m)
    lx(l, ξ) = gradient(ξ.x -> l(ξ), ξ.x) # return fn lx(ξ)
    lu(l, ξ) = gradient(ξ.u -> l(ξ), ξ.u)
    a(ξ) = lx(l, ξ)' # return lx(ξ)'
    b(ξ) = lu(l, ξ)'

    ḣ(ξ, t) = l(ξ, t)
    mx(m, ξ) = gradient(ξ.x -> m(ξ), ξ.x)' # dimensionality?
    r1(ξ) = mx(m, ξ)'
    return a, b, ḣ, r1
end

function loss_grads2(a, b, r1)
    lxx(a) = ξ -> gradient(ξ.x -> a(ξ)', ξ.x) # uh oh fns of time
    lxu(a) = ξ -> gradient(ξ.u -> a(ξ)', ξ.u)
    luu(b) = ξ -> gradient(ξ.u -> b(ξ)', ξ.u) 
    mxx(r1, ξ) = gradient(ξ.x -> r1(ξ), ξ.x)'
    P1(ξ) = mxx(r1, ξ) #TODO: Check pos def?
    return lxx(a), lxu(a), luu(b), P1
end

build_h(l, m, ξ, T) = quadgk(l(ξ), 0, T) + m(ξ(T)) # return h(ξ)

function tot_grads(a, b, Q₀, R₀, S₀)
    Dh(ξ, ζ) = a(ξ.x)' * ζ.x + b(ξ.u)' * ζ.u
    D2g(ζ) = wnorm(ζ.x, Q₀) + 2ζ.z'*S₀*ζ.x + wnorm(ζ.x, R₀)
    return Dh, D2g
end

function set_up_cost(l, m)
    # example of how to get all cost quantities
    a, b, ḣ, r1 = loss_grads1(l, m)
    lxx, lxu, luu, P1 = loss_grads2(a, b, r1)
    h(ξ, T) =  build_h(l, m, ξ, T)
    Dh, D2g = tot_grads(a, b, Q₀, R₀, S₀)
end

begin
    z = ζ.x; v = ζ.u; x = ξ.x; u = ξ.u
    gradient(x->h(?), z) + gradient(u->h(?), v)
end