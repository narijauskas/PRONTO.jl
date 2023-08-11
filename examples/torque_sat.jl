using PRONTO
using StaticArrays 
using LinearAlgebra
using MatrixEquations
include("epsdelkap.jl")
## ------------------------------- helper functions ------------------------------- ##

function skew(x)
    return [0 -x[3] x[2];
            x[3] 0 -x[1];
            -x[2] x[1] 0]
end

function Oleft(q)
    q_s=q[1]
    q_v=q[2:4]
    return [q_s -q_v';
            q_v q_s*I(3)+skew(q_v)]
end

function Oright(q)
    q_s=q[1]
    q_v=q[2:4]
    return [q_s -q_v';
            q_v q_s*I(3)-skew(q_v)]
end

function ZProj(q)
    return Oright(q)[1:4,2:4]
end

function normq(q,p)
    # Computes the Geodesic Norm distance (shortest rotation angle) between the
    # two unit quaternions q, p = [w;x;y;z]. This norm is analytically given by:
    #     d(q1,q2) = |log(q1^-1 '*' p)|
    #              = acos(2*(<q,p>)^2 - 1)
    # Where '*' is the quaternion product and <x,y> is the standard (vector) inner
    # product on R4. This norm computes the shortest rotation angle (rad) between q,p.
    
    if((size(q,1) != 4) || (size(p,1) != 4))
        @error("Input quaternions must be 4xN\n")
    end

    # Normalize (reduce numerical error)
    q_n = q ./ norm(q)
    p_n = p ./ norm(p)
    
    # Compute norm (numerically conditioned)
    d_arg = 2 .* (dot(q_n,p_n).^2) - 1
    if(abs(d_arg) - 1 > 1e2*eps())
        @error("normq returning imaginary values...")
    else
        d   = real(acos(d_arg))
    end
    
    return d
end


function quat2rotm(q)
    qs = q[1]
    qv = q[2:4]
    return (qs^2 * I(3) + 2*qs*skew(qv) + qv*qv' + skew(qv)^2)
end

function logw(q)
    # Compute the logarithm log(q) of the unit quaternion q = [q_w;q_x;q_y;q_z]
    # to return a 3x1 rotation vector
    
    if(abs(1 - norm(q)) >= 1e2*eps())
        @warn("Input is not a unit quaternion, normalizing...")
        q = q ./ norm(q);
    end
    
    q_w     = q[1];                     # Real part
    q_v     = q[2:4];                   # Imag. part
    
    nq      = norm(q);                  # Norm of quaternion
    nq_v    = norm(q_v);                # Norm of imaginary (vector) component
    phi     = atan(nq_v,q_w);           # Double rotation angle
    
    if(nq_v^4 > eps())  # Normal case
        lw = (phi/nq_v).*q_v;           # Imag part
        
    else                # Numerical ill-conditioning
        phi_sphi= 1 + (phi^2)/6;        # Taylor series exp of (x/sin(x))
        
        lw = (phi_sphi/nq).*q_v;        # Imag part
    end
    
end

function GenGuessXU(t,tf,x0,xd,J)
    # Define state components
    #  - Assumes x0 and xd are already defined in scope
    q0 = x0[1:4]    # Initial state (quaternion)
    w0 = x0[5:7]    # Initial state (angular rate)
    qd = xd[1:4]    # Target state (quaternion)

    if(norm(w0) != 0)
        @warn("Nonzero initial velocity given, guess assumes w0 = 0!\n");
    end
    
    # Choose closest representation on geodesic
    qdp     = qd
    if(normq(-qd,q0) < normq(qd,q0))
        qdp = - qd
    end

    # Compute scaled time vector 
    tau     = 0.5.*(1 - cos(pi*t./tf))          # Time-scaling (allows non-constant w_t choice), must be 0 -> 1
    dtaudt  = 0.5.*sin(π*t./tf).*(π./tf)        # Derivative function w.r.t. original time sampling
    dtau2   = 0.5.*cos(π*t./tf).*((π./tf)^2)    # 2nd derivative


    # Compute minimum geodesic on S3 w/ sampling tau
    phi = abs(2*atan(norm(q0-qdp),norm(q0 + qdp)))
    if(phi > π/2)
        # Never used as qdp choice forces this not to enter
        psi = π - phi;
        ps  = [ sin(psi - tau.*psi)./sin(psi);
                     -sin(tau.*psi)./sin(psi)]
    else
        psi = phi;
        ps  = [ sin(psi - tau.*psi)./sin(psi);
                     sin(tau.*psi)./sin(psi)]
    end
    q_t = ps[1] * q0  +  ps[2] * qdp 



    # Compute Corresponding Angular Velocity (analytic)
    # If q maps i -> b (tracks motion of body frame), we expect: expq(w/2) ~ q_d '*' (q_0*)
    #   as this quaternion represents the entire rotation
    # Since q is the OTHER way by convention, we have instead:   expq(w/2) ~ (q_0*) '*' q_d
    #   This is NOT the conjugate transform, 
    q0c   = [q0[1];-q0[2:4]]                        # Conjugate
    w_rot = 2*logw( Oleft(q0c) * qdp )              # FWD Rotation direction/angle
    w_t   = dtaudt*w_rot                            # Scaled to original time scale

    # Store trajectory results
    α_t   = [q_t;w_t]

    # Compute corresponding control inputs (analytic)
    dwt   = dtau2*w_rot                             # Compute second derivative
    μ_t   = J*dwt + skew(w_t)*J*w_t

    return α_t, μ_t
end



## --------------------- Constraints --------------------- ##
# Added for convenience in this location at present
# - should add weight defaults, deltas, kappas, and scaling rules

# Convex constraints
function cvx_constraints(x,u)
    u_max = 0.25
    w_max = 0.20

    w = x[5:7]

    c_cvx = [ (u ./ u_max).^2 .- 1;
              (w ./ w_max).^2 .- 1]

    return c_cvx
end


# Non-convex constraints
function ncvx_constraints(x,u)
    cphi_kpout = cosd(10);              # Cosine of keepout angle
    vb_satcam  = [1;1;1] ./ sqrt(3)     # orientation of camera on satellite (body frame)
    vi_sun     = [1;-1;1] ./ sqrt(3)     # orientation of sun (inertial frame)
    #vi_sun     = quat2rotm(GenGuessXU(tf/2,tf,x0,xd,J)[1][1:4]) * vb_satcam;   # Compute keepout vector to intersect geodesic
    
    q = x[1:4]

    c_ncvx = vi_sun' * quat2rotm(q) * vb_satcam - cphi_kpout

    return c_ncvx
end

# Evaluate all constraints
function constraints(x,u)
    return [cvx_constraints(x,u);
            ncvx_constraints(x,u)]
end


## ------------------------------- modeling ------------------------------- ##

@kwdef mutable struct TorqueSatC <: PRONTO.Model{7,3}
    J::SMatrix{3,3,Float64} = diagm([10.622; 10.622; 6.201])
    ε::SVector{7,Float64} = [0.1*ones(6); 1]
    δ::SVector{7,Float64} = ones(7)
    κ::Float64 = 250
end

# ε_0 = [0.1*ones(6); 1]
# ε_i = 0.21*ones(7) # fast mode
# ε_f = 2e-3.*ε_0
# δ_0 = ones(7)
# # δ_i = ?
# δ_f = 1e-12*ones(7)

f_sym = @define_f TorqueSatC begin
    q = x[1:4]
    ω = x[5:7]
    τ = u
    ω̇ = J\(τ - skew(ω)*J*ω)
    q̇ = 1/2*Oright(q)*[0;ω]
    [q̇; ω̇]
end

# Specify/Precompute problem elements to define cost/regulator
qd = [1;0;0;0]      # desired attitude
ωd = [0;0;0]        # desired angular velocity
xd = [qd; ωd]       # desired state
ud = [0;0;0]        # desired input

θ = TorqueSatC()
tf = 1.0
Qs = I(6)           # projected stage cost
Rd = I(3)           # projected input cost

# Compute running/terminal state costs using projected ARE
Mqd = [ZProj(qd)' zeros(3,3);   # projection to q-w tangent space around target
        zeros(3,4) I(3)]
Ad = PRONTO.fx(θ, xd, ud, tf)          # linearized state dynamics at target
Bd = PRONTO.fu(θ, xd, ud, tf)          # linearized input dynamics at target
And = Mqd*Ad*Mqd'                # Projected state dynamics at target
Bnd = Mqd*Bd                     # Projected input dynamics at target

Pn,_ = arec(And,Bnd*(Rd\Bnd'),Qs)
Pd = Mqd'*Pn*Mqd
Qd = Mqd'*Qs*Mqd


l_sym = @define_l TorqueSatC begin
    cc = -cvx_constraints(x,u)
    nc = -ncvx_constraints(x,u)
    c = ε.*[
        β_δ.(cc, δ[1:6])
        β_δ(σ_κ(nc,κ), δ[7])
    ]
    1/2*(x-xd)'*Qd*(x-xd) + 1/2*u'*Rd*u + sum(c)
end

@define_c TorqueSatC begin
    cc = cvx_constraints(x,u)
    nc = ncvx_constraints(x,u)
    [cc;nc]
end

@define_m TorqueSatC begin  
    1/2*(x-xd)'*Pd*(x-xd)
end

@define_Q TorqueSatC begin
    q = x[1:4]
    Mq = [ZProj(q)' zeros(3,3);
           zeros(3,4) I(3)]
    Q = Mq'*Qs*Mq
end

@define_R TorqueSatC Rd

# overwrite default behavior of Pf
function PRONTO.Pf(θ::TorqueSatC, αf, μf, tf)

    qf = αf[1:4]
    Mqf = [ZProj(qf)' zeros(3,3);
            zeros(3,4) I(3)]

    Ar = PRONTO.fx(θ, αf, μf, tf)
    Br = PRONTO.fu(θ, αf, μf, tf)
    Qr = PRONTO.Q(θ, αf, μf, tf)
    Rr = PRONTO.R(θ, αf, μf, tf)

    An = Mqf*Ar*Mqf'
    Bn = Mqf*Br
    Qn = Symmetric(Mqf*Qr*Mqf')

    Pn,_ = arec(An,Bn*(Rr\Bn'),Qn)
    Pf = Mqf'*Pn*Mqf
    return SMatrix{7,7,Float64}(Pf)
end



# must be run after any changes to model definition
# resolve_model(TorqueSatC)
PRONTO.resolve_all(TorqueSatC, f_sym, l_sym)





# using Symbolics
# θ = TorqueSat()
# x = PRONTO.symbolic(:x, 1:nx(θ))
# u = PRONTO.symbolic(:u, 1:nu(θ))
# ε = PRONTO.symbolic(:ε, 1:7)
# δ = PRONTO.symbolic(:δ, 1:7)
# κ = PRONTO.symbolic(:κ)

# cc = -cvx_constraints(x,u)
# nc = -ncvx_constraints(x,u)
# lε = ε.*[
#     β_δ.(cc, δ[1:6])
#     β_δ(σ_κ(nc,κ), δ[7])
# ]



## ------------------------------- solving ------------------------------- ##

q0 = @SVector [0;1;0;0]
w0 = @SVector [0;0;0]
x0 = [q0;w0]

t0,tf = τ = (0,60)

θ = TorqueSatC()


# # Default (final state at static equilibrium)
# α = t->xd
# μ = t->@SVector zeros(3)
J = θ.J
# Precomputed geodesic initial guess
α = t->GenGuessXU(t,tf,x0,xd,J)[1]
μ = t->GenGuessXU(t,tf,x0,xd,J)[2]


θ = TorqueSatC()
η = closed_loop(θ,x0,α,μ,τ)
@time ξ,data = pronto(θ,x0,η,τ; tol = 1e-4)

θ.ε = θ.ε*ε_i

ε_0 = [0.1*ones(6); 1]
ε_i = 0.21*ones(7) # fast mode
ε_f = 2e-3.*ε_0

while !all(θ.ε .<= ε_f)
    θ.ε = θ.ε.*ε_i # only decrement the ones that haven't satisfied the tolerance
    ξ,data = pronto(θ,x0,ξ,τ; tol = 1e-4)
end

## ------------------------------- plotting ------------------------------- ##

using CairoMakie

# plot quaternion, angular rate state elements separately
fig = Figure()
ts = 0:0.001:tf
ax = Axis(fig[1,1]; xlabel="time [s]", ylabel="quaternion")
qt = [data.ξ[end].x(t)[i] for t∈ts, i∈1:4]
foreach(i->lines!(ax, ts, qt[:,i]), 1:4)

ax = Axis(fig[2,1];xlabel="time [s]", ylabel="angular velocity [rad/s]")
wt = [data.ξ[end].x(t)[i] for t∈ts, i∈5:7]
foreach(i->lines!(ax, ts, wt[:,i]), 1:3)

# Plot control input
ax = Axis(fig[3,1]; xlabel="time [s]", ylabel="input [Nm]")
u = [data.ξ[end].u(t)[i] for t∈ts, i∈1:3]
foreach(i->lines!(ax, ts, u[:,i]), 1:3)
display(fig)

# Compute and plot quaternion norm to validate error tolerance
nqt = zeros(length(ts))
for i = 1:length(ts)
    nqt[i] = norm(qt[i,:])
end
ax = Axis(fig[4,1]; xlabel="time [s]", ylabel="quaternion norm")
lines!(ax, ts, vec(1 .- nqt))




save("torque_sat.png", fig)#https://github.com/Thomas-Dearing/ProntoDev