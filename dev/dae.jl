using DifferentialEquations
using LinearAlgebra
using SciMLBase: @def

@def x ξ[1:nx(M)]
@def u ξ[nx(M)+1]

function dae!(dξ,ξ,(M,θ,μ,α),t)
    PRONTO.f!(dξ[1:nx(M)],ξ[1:nx(M)],ξ[nx(M)+1:end],t,θ)
    dξ[nx(M)+1:end] .= μ - PRONTO.Kr()(x-α)
end


massmatrix(M) = cat(diagm(ones(nx(M))), zeros(nu(M)); dims=(1,2))


dae = 
prob_mm = ODEProblem(ODEFunction(dae!,mass_matrix=M),[x0;u0],(t0,tf),(...))
sol = solve(prob_mm,Rodas5(),reltol=1e-8,abstol=1e-8)