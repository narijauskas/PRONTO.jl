# wraps SciML interpolants with a simpler type wrapper

struct SolutionWrapper{T}
    interp
    p
    tspan::Tuple{Float64,Float64}
end

function SolutionWrapper(sol::SciMLBase.AbstractODESolution)
    SolutionWrapper{typeof(first(sol))}(sol.interp, sol.prob.p, sol.prob.tspan)
end

(sw::SolutionWrapper{T})(t,::Type{deriv}=Val{0}; idxs=nothing, continuity=:left) where {T, deriv} = sw.interp(t, idxs, deriv, sw.p, continuity)

Base.show(io::IO, sw::SolutionWrapper{T}) where {T} = println("SolutionWrapper: $T on $(sw.tspan)")



## ---------------------------- text ---------------------------- ##

struct Timeseriez{T}
    interp::SciMLBase.AbstractDiffEqInterpolation
    p::Any
    tspan::Tuple{Float64,Float64}
end

function Timeseriez(sol::SciMLBase.AbstractODESolution)
    Timeseriez{typeof(first(sol))}(sol.interp, sol.prob.p, sol.prob.tspan)
end

(X::Timeseriez{T})(t,::Type{deriv}=Val{0}; idxs=nothing, continuity=:left) where {T, deriv} = X.interp(t, idxs, deriv, sw.p, continuity)

Base.show(io::IO, X::Timeseriez{T}) where {T} = println("Timeseriez of $T on $(X.tspan)")


function Timeseriez(f, τ)
    interp = SciMLBase.LinearInterpolation(τ, map(f,τ))
    tspan = (first(τ), last(τ))
    Timeseriez{typeof(f(first(τ)))}(interp, nothing, tspan)
end