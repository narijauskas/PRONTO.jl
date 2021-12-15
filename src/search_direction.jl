using LinearAlgebra
# assume A,B, Q, R, x(t), xd(t), u(t), ud(t), f, Kᵣ

# for general cost functional:
# l, m = build_LQ_cost(ξd, Q, R, P1, T)
# a, b, ḣ, r1 = loss_grads1(l, m)
# lxx, lxu, luu, P1 = loss_grads2(a, b, r1)
# h(ξ, T) =  build_h(l, m, ξ, T)


#TODO: fix issue with differentiation -> interp ξ ?
# a, b, r₁ = PRONTO.loss_grads1(l, m, ξ, T)
# a = interp(a, T0); 
# b = interp(b, T0)
# lxx, lxu, luu, _ = PRONTO.loss_grads2(l, m, ξ, T)
# lxx = interp(lxx, T0); lxu = interp(lxu, T0); luu = interp(luu, T0)

# temporary (instead of above):

function search_direction(f, ξ, ξd, Qc, Rc, P₁, Kr, T)

    # linearizations & hessians
    A = Jx(f, ξ.x, ξ.u)
    B = Ju(f, ξ.x, ξ.u)
    fxx = Hxx(f, ξ.x, ξ.u)
    fxu = Hxu(f, ξ.x, ξ.u)
    fuu = Huu(f, ξ.x, ξ.u)

    a = t -> Qc*(ξ.x(t) - ξd.x(t)) # generalize to l_x'
    b = t -> Rc*(ξ.u(t) - ξd.u(t)) # generalize to l_u'
    
    # q(T) = r₁
    r₁ = P₁ * (ξ.x(T) - ξd.x(T))
    q = solve(ODEProblem(qstep!, r₁, (T,0), (Kr,a,b,A,B)))

    # calculate R₀,S₀,Q₀ & check if posdef
    # do we want to check posdef(Q₀)? og pronto only checks posdef(R₀)
    R₀ = t -> Rc .+ sum(map((qk,fk) -> qk*fk, q(t), fuu(t)))
    # R₀ = t -> luu(t) + sum(map((qk,fk) -> qk*fk, q(t), fuu(t)))
    
    if false && isposdef(R₀(T))
        #TODO: only do 2nd order after first iteration
        println("Using 2nd order descent")
        # Q₀ = t -> lxx(t) + sum(map((qk,fk) -> qk*fk, q(t), fxx(t)))
        # S₀ = t -> lxu(t) + sum(map((qk,fk) -> qk*fk, q(t), fxu(t)))
        Q₀ = t -> Qc .+ sum(map((qk,fk) -> qk*fk, q(t), fxx(t)))
        S₀ = t ->       sum(map((qk,fk) -> qk*fk, q(t), fxu(t)))

    else
        println("Using 1st order descent")
        # R₀ = t -> luu(t)
        # Q₀ = t -> lxx(t)
        # S₀ = t -> lxu(t)
        Q₀ = t -> Qc
        R₀ = t -> Rc
        S₀ = t -> zeros(size(Qc,1),size(Rc,2))
    end

    # do we want to recalculate P1 on each loop?
    # P₁,_ = arec(A(T), B(T), R₀(T), Q₀(T), S₀(T))
    # reuse r₁
    # P,r = solve(ODEProblem(backstep!, [P₁;r₁], (T,0), (a,b,R₀,S₀,Q₀,A,B)))
    # return K₀ = inv(R₀(t))*(S₀(t)' + B(t)'P) # instantaneous P
    # return (a,b,R₀,S₀,Q₀,A,B)
    println("solving pstep!")
    P = solve(ODEProblem(pstep!, P₁, (T,0), (a,b,R₀,S₀,Q₀,A,B)))
    println("ODE solved")

    K₀ = t -> inv(R₀(t))*(S₀(t)' .+ B(t)'P(t)) # time-variant P

    println("solving rstep!")
    r = solve(ODEProblem(rstep!, r₁, (T,0), (a,b,A,B,K₀)))
    println("ODE solved")

    # K₀ = t -> inv(R₀(t))(S₀(t)' + B(t)'P(t)) # P is a function of time
    v₀ = t -> -inv(R₀(t))*(B(t)'r(t) + b(t))

    return (A,B,K₀,v₀)
    println("solving zstep!")
    z₀ = zeros(size(ξ.x(T))...) #TODO: what is z(0)?
    z = solve(ODEProblem(zstep!, z₀, (0,T), (A,B,K₀,v₀)))
    println("ODE solved")
    v = t -> -K₀(t)z(t) + v₀(t)

    #TODO: calculate costs zn+1 and zn+2
    # maybe outside fxn?

    return Trajectory(z, v)

end


function qstep!(q̇, q, (Kr,a,b,A,B), t)
    q̇ .= -(A(t) - B(t)*Kr(t))'q - a(t) + Kr(t)'*b(t)
end

function pstep!(dP, P, p, t)
    (a,b,R₀,S₀,Q₀,A,B) = p
    #TODO: verify K₀ calculation/dimensions
    K₀ = inv(R₀(t))*(S₀(t)' .+ B(t)'P) # instantaneous P
    dP .= -A(t)'P - P*A(t) + K₀'R₀(t)K₀ + Q₀(t)
    # ṙ .= -(A(t)-B(t)K₀)'r - a(t) + K₀*b(t)
end

function rstep!(dr, r, p, t)
    (a,b,A,B,K₀) = p
    #TODO: verify K₀ calculation/dimensions    
    # dP .= -A(t)'P - P*A(t) + K₀'R₀(t)K₀ + Q₀
    dr .= -(A(t)-B(t)K₀(t))'r - a(t) + K₀(t)'b(t)
end

# function backstep!((Ṗ,ṙ), [P,r], p, t)
#     (a,b,R₀,S₀,Q₀,A,B) = p

#     K₀ = inv(R₀(t))*(S₀(t)' + B(t)'P) # instantaneous P
#     Ṗ .= -A(t)'P - P*A(t) + K₀'R₀(t)K₀ + Q₀
#     ṙ .= -(A(t)-B(t)K₀)'r - a(t) + K₀*b(t)
# end

function zstep!(dz, z, (A,B,K₀,v₀), t)
    v = -K₀(t)z + v₀(t)
    dz .= A(t)z + B(t)v
end


# Dh, D2g = tot_grads(a, b, Q₀, R₀, S₀)

