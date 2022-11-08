
using PRONTO
using FastClosures
using StaticArrays
using LinearAlgebra
using MatrixEquations

##
NX = 2
NU = 1
NΘ = 0

struct InvPend <: PRONTO.Model{NX,NU,0}
end
##

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
    

    # x_eq = zeros(nx(M))
    # u_eq = zeros(nu(M))
    
    # global fx,fu,lxx,luu,lxu
    # global lxx,luu,lxu
    # global P
    # PT,_ = arec(Ar(T), Br(T)*iRr(T)*Br(T)', Qr(T))
    p = (θ,t,x,u) -> begin
        # ξ_eq = vcat(x_eq,u_eq)
        # P,_ = arec(fx(θ,t,ξ_eq), fu(θ,t,ξ_eq)*inv(Rr(θ,t,x_eq,u_eq))*fu(θ,t,ξ_eq)', Qr(θ,t,x_eq,u_eq))
        # P,_ = arec(fx(θ,t,ξ_eq), fu(θ,t,ξ_eq), luu(θ,t,ξ_eq), lxx(θ,t,ξ_eq), lxu(θ,t,ξ_eq))
        P = [
            88.0233 39.3414;
            39.3414 17.8531;
        ]
        1/2*collect(x)'*P*collect(x)
    end

    @derive InvPend
end



##

M = InvPend()
θ = Float64[]
x0 = [2π/3;0]
u0 = [0.0]
ξ0 = [x0;u0]
ξf = [0;0;0]
t0 = 0.0; tf = 10.0

x_eq = zeros(nx(M))
u_eq = zeros(nu(M))


φg = @closure t->ξf
φ = guess_φ(M,θ,ξf,t0,tf,φg)
##
@time ξ = pronto(M,θ,t0,tf,x0,u0,φ; tol = 1e-8, maxiters=100)


##
@macroexpand PRONTO.@build InvPend (Qr(θ, t, ξ)->begin
                #= c:\Users\mantas\code\PRONTO.jl\src\PRONTO.jl:323 =#
                #= c:\Users\mantas\code\PRONTO.jl\src\PRONTO.jl:325 =#
                local (x, u) = split(InvPend(), ξ)
                #= c:\Users\mantas\code\PRONTO.jl\src\PRONTO.jl:326 =#
                (esc(:Qr))(θ, t, x, u)
            end)
    #= c:\Users\mantas\code\PRONTO.jl\src\PRONTO.jl:330 =#
    #= c:\Users\mantas\code\PRONTO.jl\src\PRONTO.jl:330 =# 