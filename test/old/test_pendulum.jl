
using PRONTO

NX = 2
NU = 1
NΘ = 0

struct InvPend <: PRONTO.Model{NX,NU,NΘ}
end


let
    g = 9.81
    L = 2
    f = (θ,t,x,u) -> [
        x[2],
        g/L*sin(x[1])-u*cos(x[1])/L
    ]
    
    Ql = I
    Rl = I
    l = (θ,t,x,u) -> 1/2*collect(x)'*Q*collect(x) + 1/2*Rl*collect(u)^2

    p = (θ,t,x,u) -> begin
        P = arec(fx(θ,t,x,u), fu(θ,t,x,u), R, Q, S)
    end

    

    Rr = (θ,t,x,u) ->
    Qr = (θ,t,x,u) ->


    @derive InvPend
end



##

M = InvPend()
x0 = [2π/3;0]
t0 = 0.0; tf = 10.0

x_eq = zeros(nx(M))
u_eq = zeros(nu(M))

φ = # simulate guess at xf,uf