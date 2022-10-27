

## ----------------------------------- dependencies ----------------------------------- ##


using PRONTO
using StaticArrays
using LinearAlgebra



NX = 4
NU = 1
NΘ = 0
struct TwoSpin <: PRONTO.Model{NX,NU,NΘ}
end

struct TwoSpinP <: PRONTO.Model{NX,NU,1}
end

# ----------------------------------- model definition ----------------------------------- ##

let
    # model dynamics
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    f = (θ,t,x,u) -> collect((H0 + u[1]*H1)*x)


    # stage cost
    Ql = zeros(NX,NX)
    Rl = [0.01]
    l = (θ,t,x,u) -> (1/2*collect(x)'*Ql*collect(x) .+ 1/2*collect(u)'*Rl*collect(u))

    # terminal cost
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    p = (θ,t,x,u) -> 1/2*collect(x)'*Pl*collect(x)

    # regulator
    Rr = (θ,t,x,u) -> diagm([1])*θ[1]
    Qr = (θ,t,x,u) -> diagm([1,1,1,1])
    # Pr(θ,t,x,u)

    @derive TwoSpinP
end

# PRONTO.Ko(M)
# PRONTO.ξ_t(M)

## ----------------------------------- tests ----------------------------------- ##

M = TwoSpinP()
θ = 1
t0 = 0.0
tf = 10.0
x0 = [0.0;1.0;0.0;0.0]
xf = [1.0;0.0;0.0;0.0]
u0 = [0.0]


#
φ = PRONTO.guess_zi(M,θ,xf,u0,t0,tf)
@time ξ = pronto(M,θ,t0,tf,x0,u0,φ)


# #Time sampling
# # macro lets you define a pair of timepoints
# @tick name # samples name_tik = @time_ns()
# @tock name # samples name_tok = @time_ns()
# @clock name # (name_tik - name_tok)
# @μs name # (name_tik - name_tok)/1e3
# @ms name # (name_tik - name_tok)/1e6



# tick(name) = esc(Symbol(String(name)*"_tick"))
# tock(name) = esc(Symbol(String(name)*"_tock"))

# macro tick(name)
#     :($(_tick(name)) = time_ns())
# end

# macro tock(name)
#     :($(_tock(name)) = time_ns())
# end

# macro clock(name)
#     tick = _tick(name)
#     tock = _tock(name)
#     ms = :(($tock - $tick)/1e6)
#     :("$($:(round($ms; digits=3))) ms")
# end