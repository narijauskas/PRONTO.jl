
function riccati!(Ṗ, P, (A,B,Q,R), t)
    Ṗ = -A(t)'P - P*A(t) + P*B(t)*inv(R)*B(t)'P - Q
end


# create an optimal LQR controller around the linearization of f(ξ)
function optKr(f, ξ, Q, R, P₁, T)
    A = Jx(f, ξ)
    B = Ju(f, ξ)
    # P₁,_ = arec(A(T), B(T)inv(R)B(T)', Q) # solve algebraic riccati eq at time T
    return P = solve(ODEProblem(riccati!, P₁, (T, 0.0), (A,B,Q,R))) # solve differential riccati
    return Kᵣ = t->inv(R)*B(t)'*P(t)
end


function ẋl!(dx, x, (f, ξ, Kr), t)
    # u = μ + Kᵣ*(α - x)
    u = ξ.u(t) + Kr(t) * (ξ.x(t) - x)
    dx .= f(x, u)
    # l̇ .= ḣ(x, t->u, t)
end

# u = μ + Kᵣ*(α - x)
project_u(ξ, x, Kᵣ) = (t) -> ξ.u(t) + Kᵣ(t) * (ξ.x(t) - x(t))

# project an arbitrary trajectory ξ onto the trajectory manifold of the system f, using the optimal controller Kᵣ, and the cost functional h
function project(ξ, f, Kᵣ, T)
    # project desired curve onto trajectory manifold using Kr
    p = (f, ξ, Kᵣ)
    prob = ODEProblem(ẋl!, ξ.x(0), (0,T), p) # IC syntax?
    x = solve(prob) 
    # u = project_u(ξ, x, Kᵣ)
    return x
end

function armijo_backstep(ξ, ζ, f, Kᵣ, (h, ḣ, Dh), (α, β)=(.7,.4))
    while γ > β^(12) # TODO: make min γ a parameter?
        # g(ξ + γ*ζ) < α*Dh(ξ, γ*ζ) ? (return γ) : (γ *= β)
        ξi = project(ξ + γ*ζ, f, Kᵣ, T)
        # h = build_h(l, m, ξi, T)
        true_cost = h(ξi)
        threshold =  α*Dh(ζ)
        true_cost < threshold ? (return γ) : (γ *= β)
    end
    γ = 0
end

fréchet() = println("je suis extra")

# user provides: Q, R, (m&l), f, ξeqb, ξd
function pronto(ξd, Q, R, (m, l), f)
    # linearize
    h = ξ -> build_h(l, m, ξ, T)
    ḣ = l
    Kᵣ = optKr(f, ξ, Q, R, P₁, T)
    ξ = project(ξd, f, Kᵣ, T)
    while γ > 0 # if keep γ as only condition, move initialization into loop?
        #TODO: is there a better way to check for convergence?
        ζ = search_direction()
        Dh = build_Dh(a, b, r1) # returns Dh(ζ)
        # check for convergence here
        γ = armijo_backstep(ξ, ζ, f, Kᵣ, (h, ḣ, Dh))
        ξ = ξ + γ*ζ
        Kᵣ = optKr(f, ξ, Q, R, P₁, T)
        ξ = project(ξ, f, Kᵣ, T) # update trajectory
        #MAYBE: print cost of that step (lxi)
    end
    return ξ, Kᵣ
end