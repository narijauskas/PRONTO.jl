using LinearAlgebra
# assume A,B, Q, R, x(t), xd(t), u(t), ud(t), f, Kᵣ

# for general cost functional:
# l, m = build_LQ_cost(ξd, Q, R, P1, T)
# a, b, ḣ, r1 = loss_grads1(l, m)
# lxx, lxu, luu, P1 = loss_grads2(a, b, r1)
# h(ξ, T) =  build_h(l, m, ξ, T)

# calculate a,b
# a = t -> Q*(x(t)-xd(t)) #TODO: l_x'
# b = t -> R*(u(t)-ud(t)) #TODO: l_u'

#TODO: fix issue with differentiation 
#TODO: interp ξ ?
a, b, r₁ = PRONTO.loss_grads1(l, m, ξ, T)
a = interp(a, T0); 
b = interp(b, T0)
lxx, lxu, luu, _ = PRONTO.loss_grads2(l, m, ξ, T)
lxx = interp(lxx, T0); lxu = interp(lxu, T0); luu = interp(luu, T0)

# temporary (instead of above):
a = t -> Qc*(ξ.x(t) - ξd.x(t))
b = t -> Rc*(ξ.u(t) - ξd.u(t))
r₁ = P₁ * (ξ.x(T) - ξd.x(T))
lxx = Qc 
luu = Rc
lxu = [1, 1]'

function qstep!(q̇, q, (Kᵣ,a,b), t)
    q̇ .= -(A(t) - B(t)*Kᵣ(t))'q - a(t) + Kᵣ(t)'*b(t)
end

# q(T) = r₁
q = solve(ODEProblem(qstep!, r₁, (T,0), (Kᵣ,a,b)))

# calculate R₀,S₀,Q₀
# Q₀ = t -> Q(t) + sum(map((qk,fk) -> qk*fk, q(t), fxx(t)))
# R₀ = t -> R(t) + sum(map((qk,fk) -> qk*fk, q(t), fuu(t)))
# S₀ = t ->        sum(map((qk,fk) -> qk*fk, q(t), fxu(t)))

R₀ = t -> luu(t) + sum(map((qk,fk) -> qk*fk, q(t), fuu(t)))
if isposdef(R₀) #TODO: only do 2nd order after first iteration
    println("Using 2nd order descent")
    Q₀ = t -> lxx + sum(map((qk,fk) -> qk*fk, q(t), fxx(t)))
    S₀ = t -> lxu + sum(map((qk,fk) -> qk*fk, q(t), fxu(t)))
else
    println("Using 1st order descent")
    R₀ = t -> luu(t)
    Q₀ = t -> lxx(t)
    S₀ = t -> lxu(t)
end

# Dh, D2g = tot_grads(a, b, Q₀, R₀, S₀)

function backstep!((Ṗ,ṙ), (P,r), p, t)
    (a,b,R₀,S₀,Q₀,A,B) = p

    K₀ = inv(R₀(t))*(S₀(t)' + B(t)'P) # instantaneous P
    Ṗ .= -A(t)'P - P*A(t) + K₀'R₀(t)K₀ + Q₀
    ṙ .= -(A(t)-B(t)K₀)'r - a(t) + K₀*b(t)
end

P₁,_ = arec(A(T), B(T), R₀(T), Q₀(T), S₀(T))
r₁ = P₁*(x(T)-xd(T))

P,r = solve(ODEProblem(backstep!, (P₁,r₁), (T,0), params))

K₀ = t -> inv(R₀(t))(S₀(t)' + B(t)'P(t)) # P is a function of time
v₀ = t -> -inv(R₀(t))(B(t)'r(t) + b(t))

function frontstep!(ż, z, (A,B,K₀,v₀), t)
    v = -K₀(t)z + v₀(t)
    ż .= A(t)z + B(t)v
end

z = solve(ODEProblem(frontstep!, 0, (0,T), params))
v = t -> -K₀(t)z(t) + v₀(t)

#TODO: zn+1 and zn+2 outside

#TODO: if Q₀, R₀, S₀ pos def, γ=1, else armijo step size

# function search_direction(ξ, ξd, f, Kᵣ, Q, R)

#     q = solve(ODEProblem(qstep!, ))

    
# end
