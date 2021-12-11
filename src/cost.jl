using QuadGK

wnorm(vec, mat) = 1/2 * vec'*mat*vec

# hard-code typical cost function and derivatives
function LQ_cost(ξ, ξd, Q, R, P1, T)
    err = ξ - ξd
    l = t -> wnorm(err.x(t), Q(t)) + wnorm(err.u(t), R(t))
    L, _ = quadgk(l, 0, T)
    L += wnorm(err.x(T), P1) # terminal cost
    return L
end

function LQ_cost_dt(ξ, ξd, Q, R, t)
    # add time derivative of LQ_cost
    err = ξ - ξd
    return wnorm(err.x(t), Q(t)) + wnorm(err.u(t), R(t))
end 

function DLQ_cost(ζ, ξ, ξd, Q, R, P1, T) 
    # add total derivative of LQ_cost
    err = ξ - ξd
    l = t -> err.x(t)' * Q * ζ.x + err.u(t)' * R * ζ.u
    L, _ = quadgk(l, 0, T)
    L += err.x(T)' * P1 * ζ.x(T) # terminal cost
    return L
end

#TODO: add generic cost functional
# should have form: l(ξ) = h(ξ, p(ξ, t), T)
# Dh(ξ, ζ)
# ḣ(ξ, t)
function frechet(h)
    # (ξ, ζ) -> 
    return Dh
end

