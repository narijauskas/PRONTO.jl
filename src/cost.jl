using QuadGK

wnorm(vec, mat) = 1/2 * vec'*mat*vec

function LQ_cost(ξ, ξd, Q, R, m, T)
    err = ξ - ξd
    l = (t) -> ( wnorm(err.x(t), Q(t)) + wnorm(err.u(t), R(t)) )
    L, _ = quadgk(l, 0, T)
    return L + m(ξ.x(T)) #TODO fix m?
end

function LQ_cost_dt() end #TODO: add time derivative of LQ_cost

function DLQ_cost() end #TODO: add total derivative of LQ_cost