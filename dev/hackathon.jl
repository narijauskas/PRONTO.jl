using JET


ξ = data.ξ[20]
Ko = data.Ko[20]
vo = data.vo[20]

PRONTO.search_direction(θ,ξ,Ko,vo,τ)
@code_warntype PRONTO.search_direction(θ,ξ,Ko,vo,τ)


x = data.ξ[1].x
u = data.ξ[1].u
z = data.ζ[1].x
v = data.ζ[1].u
γ = 1
Kr = data.Kr[1]
PRONTO.armijo_projection(θ,x0,x,u,z,v,γ,Kr,τ)