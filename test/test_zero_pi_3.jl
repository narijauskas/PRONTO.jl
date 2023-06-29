using PRONTO
using StaticArrays
using DelimitedFiles

H0 = readdlm("H0.csv", ',', ComplexF64)
H1 = readdlm("H1.csv", ',', ComplexF64)

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

# ------------------------------- 3lvl system Xgate ------------------------------- ##

@kwdef struct lvl3X <: Model{12,1,3}
    kl::Float64 # stage cost gain
    kr::Float64 # regulator r gain
    kq::Float64 # regulator q gain
end


function termcost(x,u,t,θ)
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    P = I(12)
    1/2 * collect((x-xf)')*P*(x-xf)
end


# ------------------------------- 3lvl system definitions ------------------------------- ##

function dynamics(x,u,t,θ)
    H00 = kron(I(2),H0[1:3,1:3])
    H11 = kron(I(2),H1[1:3,1:3])
    return mprod(-im*(H00+u[1]*H11))*x
end


stagecost(x,u,t,θ) = 1/2*θ.kl*collect(u')I*u 
regR(x,u,t,θ) = θ.kr*I(1)

function regQ(x,u,t,θ)
    θ.kq*I(12)
end

PRONTO.Pf(α,μ,tf,θ::lvl3X) = SMatrix{12,12,Float64}(I(12))

# ------------------------------- generate model and derivatives ------------------------------- ##

PRONTO.generate_model(lvl3X, dynamics, stagecost, termcost, regQ, regR)

## ------------------------------- demo: Simulation in 300 ------------------------------- ##

ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))

θ = lvl3X(kl=0.01, kr=1, kq=1)

t0,tf = τ = (0,300)

μ = @closure t->SVector{1}(0.4*cos((H0[3,3]-H0[1,1])*t))
φ = open_loop(θ,x0,μ,τ)
@time ξ = pronto(θ,x0,φ,τ; tol = 1e-4, maxiters = 50, limitγ = true)