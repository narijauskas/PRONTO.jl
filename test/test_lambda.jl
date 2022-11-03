

## ----------------------------------- dependencies ----------------------------------- ##


using PRONTO
using StaticArrays
using LinearAlgebra
using FastClosures



NX = 6
NU = 4
NΘ = 0
struct LambdaSys <: PRONTO.Model{NX,NU,NΘ}
end

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
           1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

function inprod(x)
    a = x[1:Int(NX/2)]
    b = x[(Int(NX/2) + 1):NX]
    P = [a*a'+b*b' -(a*b'+b*a');
         a*b'+b*a' a*a'+b*b']
    return P
end

xf = [0.0;0.0;1.0;0.0;0.0;0.0]

# ----------------------------------- model definition ----------------------------------- ##

let
    # model dynamics
    H0 = [-0.5 0 0;
             0 0 0;
          0 0 -0.5]

    H1 = [0 -0.5 0;
          -0.5 0 0;
             0 0 0]

    H2 = [0 -0.5im 0;
           0.5im 0 0;
               0 0 0]

    H3 = [0 0 0;
       0 0 -0.5;
       0 -0.5 0]

    H4 = [0 0 0;
     0 0 -0.5im;
      0 0.5im 0]

    f = (θ,t,x,u) -> collect(mprod(-1im*(H0 + u[1]*H1 + u[2]*H2 + u[3]*H3 + u[4]*H4))*collect(x))


    # stage cost
    Ql = zeros(NX,NX)
    Rl = 0.01
    l = (θ,t,x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)

    # terminal cost
    Pl = I(NX) - inprod(xf)
    p = (θ,t,x,u) -> 1/2*collect(x)'*Pl*collect(x)

    # regulator
    Rr = (θ,t,x,u) -> diagm([1,1,1,1])
    Qr = (θ,t,x,u) -> diagm([1,1,1,1,1,1])
    # Pr(θ,t,x,u)

    @derive LambdaSys
end


## ----------------------------------- tests ----------------------------------- ##

M = LambdaSys()
θ = nothing
t0 = 0.0
tf = 10.0
x0 = [1.0;0.0;0.0;0.0;0.0;0.0]
# xf = [1.0;0.0;0.0;0.0]
u0 = [0.0;0.0;0.0;0.0]
ξ0 = vcat(x0,u0)

##
φg = @closure t->[smooth(t,x0,xf,tf); 0.1*ones(nu(M))]

# φ = PRONTO.guess_zi(M,θ,xf,u0,t0,tf)
φ = guess_φ(M,θ,ξ0,t0,tf,φg)
@time ξ = pronto(M,θ,t0,tf,x0,u0,φ)


fig = Figure()
ax = Axis(fig[1,1])
foreach(1:10) do i
    lines!(ax, T, [ξ(t)[i] for t in T])
end

