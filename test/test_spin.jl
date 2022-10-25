

## ----------------------------------- dependencies ----------------------------------- ##


using PRONTO
using StaticArrays
using LinearAlgebra



NX = 4
NU = 1
NΘ = 0
struct TwoSpin <: PRONTO.Model{NX,NU,NΘ}
end

# ----------------------------------- model definition ----------------------------------- ##

let
    # model dynamics
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    f = (θ,t,x,u) -> collect((H0 + u[1]*H1)*x)


    # stage cost
    Ql = zeros(NX,NX)
    Rl = 0.01
    l = (θ,t,x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)

    # terminal cost
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    p = (θ,t,x,u) -> 1/2*collect(x)'*Pl*collect(x)

    # regulator
    Rr = (θ,t,x,u) -> diagm([1])
    Qr = (θ,t,x,u) -> diagm([1,1,1,1])
    # Pr(θ,t,x,u)

    @derive TwoSpin
end


## ----------------------------------- tests ----------------------------------- ##

M = TwoSpin()
θ = nothing
t0 = 0.0
tf = 10.0
x0 = [0.0;1.0;0.0;0.0]
xf = [1.0;0.0;0.0;0.0]
u0 = [0.0]


##
φ = PRONTO.guess_zi(M,θ,xf,u0,t0,tf)
ξ = pronto(M,θ,t0,tf,x0,u0,φ)

