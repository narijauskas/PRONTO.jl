using JET


ξ = data.ξ[20]
Ko = data.Ko[20]
vo = data.vo[20]

PRONTO.search_direction(θ,ξ,Ko,vo,τ)
@code_warntype PRONTO.search_direction(θ,ξ,Ko,vo,τ)
