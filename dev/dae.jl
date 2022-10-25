using DifferentialEquations
using LinearAlgebra
using FastClosures



function dae!(dξ,ξ,(M,θ,μ,α,P),t)
    x = @view ξ[1:nx(M)]
    u = @view ξ[nx(M)+1:end]
    dx = @view dξ[1:nx(M)]
    du = @view dξ[nx(M)+1:end]

    PRONTO.f!(dx,M,x,u,t,θ)
    du .= μ(t) - PRONTO.Kr(M,α(t),μ(t),t,θ,P(t))*(x - α(t)) - u
end



massmatrix(M) = cat(diagm(ones(nx(M))), zeros(nu(M)); dims=(1,2))


prob = ODEProblem(ODEFunction(dae!,mass_matrix=massmatrix(M)),[x0;u0],(t0,tf),(M,θ,μ,α,P))
sol = solve(prob)


# Wrapper{S...} = FunctionWrapper{BufferType(S...), Tuple{Float64}}

# α = @buffer (nx(M),) t->MVector{nx(M)}(zeros(nx(M)))

α = Buffer((@closure (buf,t)->(buf.=ones(nx(M)))), nx(M))
μ = Buffer((@closure (buf,t)->(buf.=zeros(nu(M)))), nu(M))
P = Buffer((@closure (buf,t)->(buf.=diagm(ones(nx(M))))), nx(M), nx(M))
