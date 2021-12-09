# assume Q, R, x(t), xd(t), u(t), ud(t), f, Kᵣ

# calculate a,b

a = (t)->Q*(x(t)-xd(t))
b = (t)->R*(u(t)-ud(t))

function qstep!(dx, x, p, t)
    q = x
    (Kᵣ,a,b) = p
    q̇ = -(A(t)-B(t)*Kᵣ)'q - a(t) + Kᵣ*b(t)

    dx = q̇
end

q = solve(ODEProblem(qstep!, ))
# calculate R₀,S₀,Q₀

function backstep!(dx, x, p, t)
    (P,r) = x
    (a,b,R₀,S₀,Q₀,A,B) = p

    K₀ = inv(R₀(t))*(S₀(t)' + B(t)'P)
    Ṗ = -A(t)'P - P*A(t) + K₀'R₀(t)K₀ + Q₀
    ṙ = -(A(t)-B(t)K₀)'r - a(t) + K₀*b(t)

    dx = (Ṗ,ṙ)
end


# afterwards
# v₀ = ...

# repack?
# create interpolants for r(t) & P(t)
# -> v₀/K₀

function frontstep!(dx, x, p, t)
    (A,B,r,P,a,b,R₀,S₀,Q₀) = p

    K₀ = inv(R₀(t))(S₀(t)' + B(t)'P)
    v₀ = -inv(R₀(t))(B(t)'r(t) + b(t))

    v = -K₀*z+v₀
    ż = A(t)*z + B(t)*v
    #TODO: zn+1 and zn+2
end



