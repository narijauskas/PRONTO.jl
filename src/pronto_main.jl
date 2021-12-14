function ẋl!((ẋ, l̇), (x, l), (f, ξ, Kᵣ, ḣ), t)
    # u = μ + Kᵣ*(α - x)
    u = ξ.u(t) + Kᵣ(t) * (ξ.x(t) - x)
    ẋ .= f(x, u)
    l̇ .= ḣ(x, u)
end

# u = μ + Kᵣ*(α - x)
project_u(ξ, x, Kᵣ) = (t) -> ξ.u(t) + Kᵣ(t) * (ξ.x(t) - x(t))

# project an arbitrary trajectory ξ onto the trajectory manifold of the system f, using the optimal controller Kᵣ, and the cost functional h
function project(ξ, f, Kᵣ, ḣ, T)
    # project desired curve onto trajectory manifold using Kr
    p = (f, ξ, Kᵣ, ḣ)
    prob = ODEProblem(ẋl!, (ξ.x(0), 0), (0,T), p) # IC syntax?
    x,l = solve(prob) # output syntax?
    u = project_u(ξ, x, Kᵣ)
    return Trajectory(x, u), l
end

function armijo_backstep(ξ, ζ, f, Kᵣ, (h, ḣ, Dh), (α, β)=(.7,.4))
    while γ >= γ^(12) # TODO: make min β a parameter?
        # g(ξ + γ*ζ) < α*Dh(ξ, γ*ζ) ? (return γ) : (γ *= β)
        ξi = project(ξ + γ*ζ, f, Kᵣ, ḣ, T)
        # h = build_h(l, m, ξi, T)
        true_cost = h(ξi)
        threshold =  α*Dh(ζ)
        true_cost < threshold ? (return γ) : (γ *= β)
    end
    γ = 0
end

fréchet() = println("je suis extra")

# user provides: Q, R, (m&l), f, ξeqb, ξd
function pronto()
    # linearize
    
    ḣ = l
    Kᵣ = optKr(A, B, Q, R, T)
    ξ, l = project(ξd, f, Kᵣ, ḣ, T)
    while γ > 0 # if keep γ as only condition, move initialization into loop?
        #TODO: is there a better way to check for convergence?
        ζ = search_direction()
        Dh = build_Dh(a, b)
        γ = armijo_backstep(ξ, ζ, f, Kᵣ, (h, ḣ, Dh))
        ξ = ξ + γ*ζ
        Kᵣ = optKr(A, B, Q, R, T)
        ξ, lxi = project(ξ, f, Kᵣ, ḣ, T) # update trajectory
        #MAYBE: print cost of that step (lxi)
    end
    return ξ, Kᵣ
end