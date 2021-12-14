function ẋl!((ẋ, l̇), (x, l), (f, ξ, Kᵣ, ḣ), t)
    # u = μ + Kᵣ*(α - x)
    u = ξ.u(t) + Kᵣ(t) * (ξ.x(t) - x)
    ẋ = f(x, u)
    l̇ = ḣ(x, u)
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

function armijo_backstep(ξ, ζ, g, Dh, (α, β)=(.7,.4))
    while γ > .01 # TODO: make min β a parameter?
        g(ξ + γ*ζ) < α*Dh(ξ, γ*ζ) ? (return γ) : (γ *= β)
        # true_cost = g(ξ + γ*ζ)
        # threshold =  α*Dh(ξ, γ*ζ) # maybe a clever way to def as α*Dh(ξ) /dot γ*ζ 
        # true_cost < threshold ? (return γ) : (γ *= β)
    end
    γ = 0
end

fréchet() = println("je suis extra")
#TODO: implicitly derive wrt vars and combine as anonymous f(t)

# user provides: Q, R, (m&t), f, ξeqb, ξd
function pronto()
    # linearize
    

    #TODO: build cost functional (m,l)->h
    #TODO: differentiate h->ḣ
    Kᵣ = optKr(A, B, Q, R, T)
    ξ, l = project(ξd, f, Kᵣ, ḣ, T)
    while γ > 0 # if keep γ as only condition, move initialization into loop?
        #TODO: is there a better way to check for convergence?
        ζ = search_direction()
        γ = stepsize(ξ) #TODO: move into search_direction? then can check posdef q
        ξ = ξ + γ*ζ
        Kᵣ = optKr(A, B, Q, R, T)
        ξ, lxi = project(ξ, f, Kᵣ, ḣ, T) # update trajectory
        #MAYBE: print cost of that step (lxi)
    end
    return ξ, Kᵣ
end