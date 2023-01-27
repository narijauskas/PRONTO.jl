using PRONTO
using StaticArrays
using LinearAlgebra

function mprod(x)
    Re = I(2)  
    Im = [0 -1;
        1 0]   
    M = kron(Re,real(x)) + kron(Im,imag(x));
    return M   
end

function inprod(x)
    i = Int(length(x)/2)
    a = x[1:i]
    b = x[i+1:end]
    P = [a*a'+b*b' -(a*b'+b*a');
        a*b'+b*a' a*a'+b*b']
    return P
end

NX = 6
NU = 4
NΘ = 2
struct Lambda <: PRONTO.Model{NX,NU,NΘ}
    kr::Float64
    kq::Float64
end


# ----------------------------------- model definition ----------------------------------- ##
function dynamics(x,u,t,θ)
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

    return mprod(-1im*(H0 + u[1]*H1 + u[2]*H2 + u[3]*H3 + u[4]*H4))*x
end

Rreg(x,u,t,θ) = θ[1]*I(NU)

function Qreg(x,u,t,θ)
    x_re = x[1:3]
    x_im = x[4:6]
    ψ = x_re + im*x_im
    θ.kq*mprod(I(3) - ψ*ψ')
end

function stagecost(x,u,t,θ)
    Rl = [0.01;;]
    1/2 * Rl * collect(u')*u
end

function termcost(x,u,t,θ)
    xf = [0;0;1;0;0;0]
    Pl = I(6)
    1/2*collect((x-xf)')*Pl*(x-xf)
end

PRONTO.generate_model(Lambda, dynamics, stagecost, termcost, Qreg, Rreg)

# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::Lambda) = SMatrix{6,6,Float64}(I(6) - inprod(α))

##

# ----------------------------------- tests ----------------------------------- ##θ = Lambda(1,1)
τ = t0,tf = 0,5

x0 = @SVector [1.0, 0.0, 0.0, 0.0, 0.0,0.0]
xf = @SVector [0.0, 0.0, 1.0, 0.0, 0.0,0.0]
# u0 = [0.1]

# smooth(t, x0, xf, tf) = @. (xf - x0)*(tanh((2π/tf)*t - π) + 1)/2 + x0
# μ = @closure t->u0*sin(t)
# α = @closure t->smooth(t, x0, xf, tf)
# φ = PRONTO.Trajectory(θ,α,μ);

μ = @closure t->SVector{4}(0.1*ones(4))
φ = open_loop(θ,x0,μ,τ)

# μ = @closure t->SizedVector{1}(u0)
# φ = open_loop(θ,xf,μ,τ) # guess trajectory
ξ = pronto(θ,x0,φ,τ) # optimal trajectory


##
using MAT
