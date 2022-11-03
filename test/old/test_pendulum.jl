
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
        g/L*sin(x[1])-u[1]*cos(x[1])/L
    ]
    
    Ql = I
    Rl = I
    l = (θ,t,x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
    
    Rr = (θ,t,x,u) -> diagm([1e-3])
    Qr = (θ,t,x,u) -> diagm([10, 1])
    

    x_eq = zeros(nx(M))
    u_eq = zeros(nu(M))
    
    # global fx,fu,lxx,luu,lxu
    # global lxx,luu,lxu
    # PT,_ = arec(Ar(T), Br(T)*iRr(T)*Br(T)', Qr(T))
    p = (θ,t,x,u) -> begin
        ξ_eq = vcat(x_eq,u_eq)
        # P,_ = arec(fx(θ,t,ξ_eq), fu(θ,t,ξ_eq)*inv(Rr(θ,t,x_eq,u_eq))*fu(θ,t,ξ_eq)', Qr(θ,t,x_eq,u_eq))
        P,_ = arec(fx(θ,t,ξ_eq), fu(θ,t,ξ_eq), luu(θ,t,ξ_eq), lxx(θ,t,ξ_eq), lxu(θ,t,ξ_eq))

        1/2*collect(x)'*P*collect(x)
    end

    @derive InvPend
end



##

M = InvPend()
x0 = [2π/3;0]
t0 = 0.0; tf = 10.0

x_eq = zeros(nx(M))
u_eq = zeros(nu(M))

φ = # simulate guess at xf,uf