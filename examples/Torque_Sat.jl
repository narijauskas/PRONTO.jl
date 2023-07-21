using PRONTO
using PRONTO: SymModel
using StaticArrays, LinearAlgebra
using MatrixEquations

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

## ------------------------------- modeling ------------------------------- ##

@kwdef struct TorqueSat <: PRONTO.Model{7,3}
    J::SMatrix{3,3,Float64} = diagm([10.622; 10.622; 6.201])
end


@define_f TorqueSat begin
    q = x[1:4]
    ω = x[5:7]
    τ = u
    ω̇ = J\(τ - skew(ω)*J*ω)
    q̇ = 1/2*Oright(q)*[0;ω]
    return [q̇; ω̇]
end

# Specify/Precompute problem elements to define cost/regulator
qd = [1;0;0;0]      # desired attitude
ωd = [0;0;0]        # desired angular velocity
xd = [qd; ωd]       # desired state
ud = [0;0;0]
θ = TorqueSat()
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



@define_l TorqueSat begin
    1/2*(x-xd)'*Qd*(x-xd) + 1/2*u'*Rd*u 
end


@define_m TorqueSat begin  
    1/2*(x-xd)'*Pd*(x-xd)
end

@define_Q TorqueSat begin
    q = x[1:4]
    Mq = [ZProj(q)' zeros(3,3);
           zeros(3,4) I(3)]
    Q = Mq'*Qs*Mq
end

@define_R TorqueSat Rd

# overwrite default behavior of Pf
function PRONTO.Pf(θ::TorqueSat,αf,μf,tf)

    q = αf[1:4]
    Mqf = [ZProj(q)' zeros(3,3);
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

αf = [0,1,0,0,0,0,0]
μf = [0,0,0]

q = αf[1:4]
Mqf = [ZProj(q)' zeros(3,3);
zeros(3,4) I(3)]

Ar = PRONTO.fx(θ, αf, μf, tf)
Br = PRONTO.fu(θ, αf, μf, tf)
Qr = PRONTO.Q(θ, αf, μf, tf)
Rr = PRONTO.R(θ, αf, μf, tf)

An = Mqf*Ar*Mqf'
Bn = Mqf*Br
Qn = Mqf*Qr*Mqf'
Pn,_ = arec(An,Bn*(Rr\Bn'),Qn)
# must be run after any changes to model definition
resolve_model(TorqueSat)

# runtime plots
# PRONTO.runtime_info(θ::TorqueSat, ξ; verbosity=1) = verbosity >= 1 && println(preview(ξ.u, 1))


## ------------------------------- demo: eigenstate 1->2 in 10s ------------------------------- ##


x0 = @SVector [0;1;0;0;0;0;0]
t0,tf = τ = (0,67)


θ = TorqueSat()
α = t->x0
μ = t->@SVector zeros(3)
η = closed_loop(θ,x0,α,μ,τ)
ξ,data = pronto(θ,x0,η,τ; tol = 1e-4, maxiters = 50, verbosity=2)