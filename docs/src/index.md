# PRONTO.jl
Hello! Welcome to the julia implementation of the **PR**ojection-**O**perator-Based **N**ewton’s Method for **T**rajectory **O**ptimization (PRONTO). PRONTO is a Newton-based method for solving trajectory optimization problems
```math
\begin{array}{rl}
\min&\displaystyle  m(x(T)) + \int^T_0\!\!\! l(x(t),u(t),t) dt \\ 
\mathrm{s.t.} & \dot{x} = f(x,u,t), \qquad ~x(0) = x_0,
\end{array}
```
where $t\in[0,T]$ is the time variable, $x\in\mathbb R^n$ and $u\in\mathbb R^m$ are the state and input vectors, $x_0\in\mathbb R^n$ is the initial condition, $f:\mathbb R^n\!\times\!\mathbb R^m\!\times\![0,T]\to\mathbb R^n$ is the dynamic model, and $l:\mathbb R^n\!\times\!\mathbb R^m\!\times\![0,T]\to\mathbb R$ and $m:\mathbb R^n\to\mathbb R$ are the incremental and terminal costs, respectively.
The key element of PRONTO is the _projection operator_, which tranforms any pair of state and input curves $[\alpha(t),\mu(t)]$ into a trajectory $[x(t),u(t)]$ that satisfies $\dot x = f(x,u,t)$ and $x(0)=x_0$. This is achieved by solving the differential equation
```math
\begin{cases}
\dot{x} = f(x,u,t), \qquad x(0) = x_0,\\
u=\mu-K_r(t)(x-\alpha),
\end{cases} 
```
where $K_r(t)$ is a time-varying feedback gain used to stabilize the trajectory $[x(t),u(t)]$ around the state and input curves $[\alpha(t),\mu(t)]$. To compute $K_r(t)$, PRONTO.jl solves the Differential Riccati Equation
```math
\begin{cases}
-\dot{P}_r = A_\eta(t)^\top P_r+P_rA_\eta(t)-K_r^\top R_r(t) K_r + Q_r(t), \qquad P(T) = P_T,\\
~~K_r=R_r(t)^{-1}B_\eta(t)^\top P_r,
\end{cases} 
```
where $A_\eta(t),B_\eta(t)$ are the Jacobians of the system dynamics, linearized around $[\alpha(t),\mu(t)]$. To use PRONTO, the user must provide suitable regulator matrices $Q_r(t),R_r(t)$. If the user does not provide a terminal condition $P_T$, PRONTO will generate one by solving the Algebraic Riccati Equation for the linearized system. 

**NOTE:** Since $K_r(t)$ plays a vital role in stabilizing the solution estimates, the choice of $Q_r(t),R_r(t)$ and $P_T$ is crucial to the covergence of PRONTO, especially for systems with unstable dynamics.

## Double Integrator
Our first example shows how to employ PRONTO.jl in the familiar context of linear time-invariant systems. To this end, consider the well-known optimal control problem
```math
\begin{array}{rl}
\min& \displaystyle\frac{1}{2}\|x(T)\|^2_P + \frac{1}{2}\int^T_0\!\!\! \|x(t)\|^2_Q + \|u(t)\|^2_R \:dt \\[8pt]
\mathrm{s.t.} & \dot{x} = Ax + Bu, \qquad x(0) = x_0.
\end{array}
```
Clearly, this problem can (and should) be solved using a standard Linear-Quadratic Regulator (LQR). However, this is a PRONTO.jl tutorial, so that's what we'll be using instead. We begin by loading some dependencies
```julia
using PRONTO
using Base: @kwdef

using LinearAlgebra
using MatrixEquations
using StaticArrays
```
The first two packages (`PRONTO` and `Base: @kwdef`) are always required to deploy PRONTO.jl. All other packages are useful for this example, but are not strictly necessary in general.

To initialize PRONTO, we first need to define our model. Since this example features a double integrator, we name our model `DoubleInt` and specify that it has a $2$ states and $1$ control input. We also choose to leave the weight matrices $R$, $Q$, and $P$, as undefined *model parameters*.
```julia
@kwdef struct DoubleInt <: Model{2,1}
    R::Float64 
    Q::SMatrix{2,2,Float64}
    P::SMatrix{2,2,Float64}
end
```
Next, we define the dynamic model of the double integrator
```math
f(x,u,t) = \begin{bmatrix} 0 & 1\\ 0 & 0 \end{bmatrix}x + \begin{bmatrix} 0\\1 \end{bmatrix}u.
```
```julia
A = [0 1; 0 0]
B = [0; 1]

@define_f DoubleInt A*x + B*u[1]
``` 
Here, the matrices $A$ and $B$ are hard-coded into the PRONTO model and cannot be changed without redefining the model from scratch. This is deemed acceptable since we're not interested in changing our system.

Next, we define the incremental and terminal cost of our model
```math
\begin{array}{rl}
l(x,u,t) = &\displaystyle\!\!\!\!\frac{1}{2}x^\top Qx + \frac{1}{2}u^\top Ru\\[8pt]
m(x)=&\displaystyle\!\!\!\!\frac{1}{2}x^\top Px
\end{array}
```
```julia
@define_l DoubleInt 1/2*R*u[1]^2 + 1/2*x'*Q*x
@define_m DoubleInt 1/2*x'*P*x
``` 
Here, the matrices $R$, $Q$, and $P$ do not need to be specified: PRONTO already knows that the user will provide them as parameters.

Finally, we need to provide the weight matrices for the projection operator. Since this example is too simple for these matrices to have a noticeable impact, we opt for the simplest choice: $R_r(t)=1$ and $Q_r(t)=I_2$.
```julia
@define_Rr DoubleInt I(1)
@define_Qr DoubleInt I(2)
```
We are now ready to build our model!
```julia
resolve_model(DoubleInt)
```
This command uses symbolic calculations to determine all the necessary Gradients, Jacobians, and Hessians used by PRONTO. You can track the status in the Terminal, but this step is generally very quick.

In preparation of solving the optimal control problem, it is now time to define all the model parameters that were not hard-coded into the model. For our purposes, we'll select
```math
R=0.04,\qquad \qquad Q=\begin{bmatrix} 1 & 0\\0 & 0 \end{bmatrix}
```
and $P$ equal to the solution to the Algebraic Riccati Equation
```math
A^{\top}P + PA - PBR^{-1}B^{\top}P + Q = 0.
``` 
```julia
R = 0.04
Q = diagm([1.0, 0.0])
P = arec(A,B,R*I,Q)[1]
```
The problem is ready to be solved! Given the initial condition $x_0=[2,0]^{\top}$ and time horizon $T=2$, we pick a guess input $\mu=0$ to start the solver; the tolerance `tol` is set to be $10^{-6}$.
```julia
θ = DoubleInt(R, Q, P) 
τ = t0,tf = 0,2
x0 = @SVector [2,0]

μ = t->[0]

η = open_loop(θ,x0,μ,τ)

ξ,data = pronto(θ,x0,η,τ; tol=1e-6);
```

We now visualize the solution using `GLMakie`
```julia
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "position [m]")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "velocity [m/s]")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "acceleration [m/s²]")

lines!(ax1, ts, [ξ.x(t)[1] for t in ts], color = :blue, linewidth = 2)
lines!(ax2, ts, [ξ.x(t)[2] for t in ts], color = :green, linewidth = 2)
lines!(ax3, ts, [ξ.u(t)[1] for t in ts], color = :red, linewidth = 2)

display(fig)
```
![image description](./double_int.png)

## Inverted Pendulum
This example showcases PRONTO's ability to:
1) steer the system to an **unstable** equilibrium point, 
2) handle **non-convex** cost functions,
3) use the **desired target** as an initial guess. 

We first load a few dependencies
```julia
using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
```

To build our OCP, we decide to name our model `InvPend`. The `{2,1}` captures the fact that we have two states, $x\in\mathbb R^2$, and one control input, $u \in \mathbb R^1$. The parameters of this model are the length of the pendulum `L`, the gravitional acceleration `g`, and the control effort penalty `ρ`. We provide nominal values for each parameter.
```julia
@kwdef struct InvPend <: Model{2,1} 
    L::Float64 = 2 
    g::Float64 = 9.81 
    ρ::Float64 = 1
end
```

Consider the dynamics of an inverted pendulum
```math
f(x,u,t)= \begin{bmatrix}x_2 \\\frac{g}{L}\sin{x_1} - \frac{u}{L}\cos{x_1}\end{bmatrix},
```
where $x_1$ is the angular position, $x_2$ is the angular velocity, and $u$ is the horizontal acceleration of the fulcrum.

To steer the pendulum to the upright position, we define the terminal cost
```math
m(x) = 1-\cos(x_1)+\tfrac12x_2^2.
```
This non-convex function is zero if and only if $x=[2k\pi~~0]^\top$, with $k\in\mathbb Z$.

To limit the control effort during the transient $[0,T]$, we define the incremental cost
```math
l(x) = \tfrac12\rho u^2,
```
where $\rho>0$ is an additional parameter of the OCP.

Now we can define the dynamics $f(x,u,t)$, the incremental cost $l(x,u,t)$, and the terminal cost $m(x)$.
```julia
@define_f InvPend [ 
    x[2], 
    g/L*sin(x[1])-u[1]*cos(x[1])/L,
    ]
@define_l InvPend 1/2*ρ*u[1]^2
@define_m InvPend 1-cos(x[1])+x[2]^2/2
```
We must now select the LQR matrices used by the projection operator. Since the linearized system is always controllable, we limit ourselves to choosing `Qr` and `Rr`. By doing so, we allow PRONTO to automatically compute the terminal conditions `PT` by linearizing the dynamics around $x(T)$ and solving the Algebraic Riccati Equation.
```julia
@define_Qr InvPend diagm([10, 1])
@define_Rr InvPend diagm([1e-3])
```
Note that, since the target equilibrium is unstable, we selected a very small input penalty $R_r$. This ensures that the regulator gain $K_r(t)$ will prioritize the angular position error (which has the highest cost) when updating the solution estimate.

The last step in the problem definition is to call `resolve_model` to instantiate PRONTO. Since we're interested in seeing how the solution estimate $x(t)$ changes from one iteration to the next, we also ask PRONTO to output an ascii plot of `ξ.x` after every iteration.
```julia
resolve_model(InvPend)
PRONTO.preview(θ::InvPend, ξ) = ξ.x
```
We are now ready to solve our OCP! To do so, we load the parameters `θ`, the time horizon `τ`, and the inital condition `x0`. Here, we will be using the default value of all our parameters (defined in `InvPend`). If we wanted to solve our OCP on Mars, we could instead call `θ = InvPend(g=3.71,ρ=10)`, where we've increased the input penalty to account for the reduced gravity.
```julia
θ = InvPend() 
τ = t0,tf = 0,10
x0 = @SVector [2π/3;0]
```
Next, we need to provide PRONTO with an initial guess. Since we wish to steer the pendulum to the upright position, we use the **desired** equilibrium as an initial guess. Therefore, our initial state and input estimates are $\alpha(t)=[0~~0]^\top$ and $u(t)=0$, respectively, $\forall t\in[0,T]$. Note that this initial guess is not a suitable solution because $\alpha(0)\neq x_0$.
```julia
μ = t->[0]
α = t->[0;0]
```
We then call the projection operator to turn our guess `α`,`μ` into a trajectory `η`. This step ensures (among other things) that the error `η.x(0)-x0` is within machine tolerance.
```julia
η = closed_loop(θ,x0,α,x,τ)
```
It is now time to call PRONTO and solve our OCP to a tolerance of $10^{-3}$
```julia
ξ,data = pronto(θ,x0,η,τ; tol=1e-3);
```
Now, we visualize the solution using `GLMakie`.
```julia
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time [s]", ylabel = "angular position [rad]")
ax2 = Axis(fig[2,1], xlabel = "time [s]", ylabel = "angular velocity [rad/s]")
ax3 = Axis(fig[3,1], xlabel = "time [s]", ylabel = "control input [m/s^2]")

lines!(ax1, ts, [ξ.x(t)[1] for t in ts], color = :blue, linewidth = 2)
lines!(ax2, ts, [ξ.x(t)[2] for t in ts], color = :green, linewidth = 2)
lines!(ax3, ts, [ξ.u(t)[1] for t in ts], color = :red, linewidth = 2)

display(fig)
```
![image description](./inv_pend_2.png)

## Qubit: State to State Transfer
We consider the Schrödinger equation
```math
i|\dot{\psi}(t)\rangle = (\mathcal{H_0} + u(t)\mathcal{H_1})|\psi(t)\rangle,
```
where $\mathcal{H_0} = \sigma_z = \begin{bmatrix}0 & 1 \\1 & 0\end{bmatrix}$, and $\mathcal{H_1} = \sigma_y = \begin{bmatrix}0 & -i \\i & 0\end{bmatrix}$ are the Pauli matrices. The real control input $u(t)$ drives the system between 2 qubit states $|0\rangle$ and $|1\rangle$, which are the two eigenstates of the free Hamiltonian $\mathcal{H_0}$. We wish to find the optimal control input $u^{\star}(t)$ that performs the state-to-state transfer from $|0\rangle$ to $|1\rangle$. To do this, we will solve:
```math
\min m(x(T)) + \int^T_0 l(x(t),u(t),t) dt \\s.t. \quad \dot{x} = f(x,u,t), x(0) = x_0,
```
First, we load the usual dependencies:
```julia
using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
```
Note that $|\psi \rangle$ is a $2 \times 1$ complex vector, and we wish to have the state vector $x$ in the real form. We can define our state vector 
```math
x = \begin{bmatrix}Re(|\psi\rangle)\\Im(|\psi\rangle) \end{bmatrix},
``` 
which in this case is a $4 \times 1$ vector of real numbers. Moreover, any complex square matrix $\mathcal{H}$ can be represented in its real form:
```math
H = \begin{bmatrix}Re(\mathcal{H}) & -Im(\mathcal{H}) \\Im(\mathcal{H}) & Re(\mathcal{H})\end{bmatrix}.
```
We can then convert the the Schrödinger equation $i|\dot{\psi}(t)\rangle = (\mathcal{H_0} + \phi(t)\mathcal{H_1})|\psi(t)\rangle$ into the nonlinear system 
```math
\dot{x}(t) = H(u)x = \begin{bmatrix}0 & 0 & 1 & 0 \\0 & 0 & 0 & -1\\-1 & 0 & 0 & 0\\0 & 1 & 0 & 0\end{bmatrix}x + u\begin{bmatrix}0 & -1 & 0 & 0 \\1 & 0 & 0 & 0\\0 & 0 & 0 & -1\\0 & 0 & 1 & 0\end{bmatrix}x.
```

We decide to name our model `Qubit`, where `{4,1}` represents the 4 state vector $x (|\psi\rangle)$, and the single input $u (\phi)$. For this example, our parameter is `kl`, which is a scalar that penalizes the control effort.
```julia
@kwdef struct Qubit <: Model{4,1}
    kl::Float64 = 0.01
end
```
First, we can define our dynamics $f$
```julia
@define_f Qubit begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end
```
For our incremental cost $l$, we simpy penalize the control effort $u$
```julia
@define_l Qubit begin
    1/2*u'*kl*u
end
```
For this example, the control objective is to steer the system from the initial state $|0\rangle = [1, 0]^T$ to the target state $|1\rangle = [0, 1]^T$. We can then define our terminal cost function $m$ as 
```math
m(x(T)) = \frac{1}{2} x^\top(T)Px(T), \qquad \mathrm{with}~P=\begin{bmatrix}1 & 0 & 0 & 0 \\0 & 0 & 0 & 0\\0 & 0 & 1 & 0\\0 & 0 & 0 & 0\end{bmatrix}
```
to penilize both real and imaginal parts for $|0\rangle$.
```julia
@define_m Qubit begin
    Pl = [1 0 0 0;0 0 0 0;0 0 1 0;0 0 0 0]
    1/2*x'*Pl*x
end
```

For this example, a Linear-Quadratic Regulator (LQR) is used. Note that, since the linearized system is *not* controllable, we need to provide a value for the terminal cost $P_T$:
```math
R_r(t) = I_1,\\Q_r(t) = I_4 ,\\P_r(T) = I_4.
``` 
```julia
@define_Qr Qubit I(4)
@define_Rr Qubit I(1)
PRONTO.Pf(θ::Qubit, αf, μf, tf) = SMatrix{4,4,Float64}(I(4))
```
Last we compute the Lagrange dynamics $L = l + \lambda^Tf$.
```julia
resolve_model(Qubit)
```
We now can solve the OCP! This time, we assume our guess input $\mu(t)=0.4\sin{t}$ and initialize our solver by computing the open loop system `open_loop`.
```julia
θ = Qubit() # instantiate a new model
τ = t0,tf = 0,10 # define time domain
x0 = @SVector [1.0, 0.0, 0.0, 0.0] # initial state
xf = @SVector [0.0, 1.0, 0.0, 0.0] # final state
μ = t->SVector{1}(0.4*sin(t)) # open loop input μ(t)
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory
```
We then visualize the solution using `GLMakie`
```julia
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "quantum state")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "control input")

lines!(ax1, ts, [ξ.x(t)[1] for t in ts], linewidth = 2, label = "Re(ψ1)")
lines!(ax1, ts, [ξ.x(t)[2] for t in ts], linewidth = 2, label = "Re(ψ2)")
lines!(ax1, ts, [ξ.x(t)[3] for t in ts], linewidth = 2, label = "Im(ψ1)")
lines!(ax1, ts, [ξ.x(t)[4] for t in ts], linewidth = 2, label = "Im(ψ2)")
fig[1, 2] = Legend(fig, ax1)
lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[3]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[4]^2 for t in ts], linewidth = 2, label = "|1⟩")
fig[2, 2] = Legend(fig, ax2)
lines!(ax3, ts, [ξ.u(t)[1] for t in ts], linewidth = 2)


display(fig)
```
![image description](./2Spin.png)

## Qubit: Pauli X Gate
We consider a 3-level fluxionium qubit, whose Hamiltonian can be written as
```math
H(u)=H_0 + uH_{\text{drive}} = \begin{bmatrix}0 & 0 & 0 \\0 & 1.0 &0\\0 & 0 & 5.0\end{bmatrix} + u\begin{bmatrix}0 & 0.1 & 0.3 \\0.1 & 0 & 0.5\\0.3 & 0.5 & 0\end{bmatrix},
```
where $H_0$ is the free Hamiltonian, $H_{\text{drive}}$ is the control Hamiltonian, and $u(t)$ is the control input. We wish to find the optimal control input $u^{\star}(t)$ that performs the X gate for this qubit, that is, $u^{\star}(t)$ steers $|0\rangle=[1,0,0]^{\top}$ to $|1\rangle=[0,1,0]^{\top}$, while simultaneously steers $|1\rangle$ to $|0\rangle$. Meanwhile, we wish to aviod the undesirade state, which is the $|2\rangle$ state of the system.

First, we load some dependencies:
```julia
using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef
```
Then we define a helper function `mprod` converting a complex matrix to its real representation  
```julia
function mprod(x)
    Re = I(2)
    Im = [0 -1;
          1 0]
    M = kron(Re,real(x)) + kron(Im,imag(x))
    return M
end
```

We decide to name our model `XGate3`, where `{12,1}` represents the 12 states vector $x ([|\psi_1\rangle, |\psi_2\rangle]^{\top})$, and the single input $u$. For this example, our parameters are `kl`, which is a scalar that penilize the control effort, and `kq`, which is a scalar that penilize the undesirade population.
```julia
@kwdef struct XGate3 <: PRONTO.Model{12,1}
    kl::Float64 = 0.01
    kq::Float64 = 0.5
end
```
First, we can define our dynamics $f$
```julia
@define_f XGate3 begin
    E0 = 0.0
    E1 = 1.0
    E2 = 5.0
    H0 = diagm([E0, E1, E2])
    H00 = kron(I(2),H0)
    a1 = 0.1
    a2 = 0.5
    a3 = 0.3
    Ω1 = a1 * u[1]
    Ω2 = a2 * u[1]
    Ω3 = a3 * u[1]
    H1 = [0 Ω1 Ω3; Ω1 0 Ω2; Ω3 Ω2 0]
    H11 = kron(I(2),H1)
    return 2 * π * mprod(-im * (H00 + H11)) * x
end
```
For our incremental cost $l$, we penilize both the control effort $u$ and undesirade third state
```julia
@define_l XGate3 begin
    kl/2*u'*I*u + kq/2*x'*mprod(diagm([0,0,1,0,0,1]))*x
end
```
For this example, the control objective is to steer the system from the $|0\rangle = [1, 0, 0]^{\top}$ state to the target state $|1\rangle = [0, 1, 0]^{\top}$, while simultaneously steers $|1\rangle$ to $|0\rangle$. We can then define our terminal cost function $m$ as 
```math
m(x(T)) = \|\psi_1(T)-|1\rangle\|^2 + \|\psi_2(T)-|0\rangle\|^2.
```

```julia
@define_m XGate3 begin
    ψ1 = [1;0;0]
    ψ2 = [0;1;0]
    xf = vec([ψ2;ψ1;0*ψ2;0*ψ1])
    return 1/2*(x-xf)'*I(12)*(x-xf)
end
```

For this example, a Linear-Quadratic Regulator (LQR) is used and designed in this way:
```math
R_r(t) = I,\\Q_r(t) = I ,\\P_r(T) = Q_r(T) = I.
``` 
```julia
@define_Qr XGate3 I(12)
@define_Rr XGate3 I(1)
PRONTO.Pf(θ::XGate3,α,μ,tf) = SMatrix{12,12,Float64}(I(12))
```
Last we compute the Lagrange dynamics $L = l + \lambda^Tf$.
```julia
resolve_model(InvPend)
```
We now can solve the OCP! This time, we assume our guess input $\mu(t)=\frac{\pi}{T}e^{\frac{-(t-T/2)^2}{T^2}}\cos{(2\pi t)}$ and initialize our solver by computing the open loop system.
```julia
θ = XGate3()
τ = t0,tf = 0,10
ψ1 = [1;0;0]
ψ2 = [0;1;0]
x0 = SVector{12}(vec([ψ1;ψ2;0*ψ1;0*ψ2]))
μ = t->SVector{1}((π/tf)*exp(-(t-tf/2)^2/(tf^2))*cos(2*π*1*t))
η = open_loop(θ, x0, μ, τ) # guess trajectory
ξ,data = pronto(θ, x0, η, τ;tol=1e-4); # optimal trajectory
```
Finally, we visualize the solution using `GLMakie`. We can see from the result that the same control input steers $|0\rangle$ to $|1\rangle$ and $|1\rangle$ to $|0\rangle$, while $|2\rangle$ stays low during the whole process. 
```julia
using GLMakie

fig = Figure()
ts = range(t0,tf,length=1001)
ax1 = Axis(fig[1,1], xlabel = "time", ylabel = "control input")
ax2 = Axis(fig[2,1], xlabel = "time", ylabel = "population")
ax3 = Axis(fig[3,1], xlabel = "time", ylabel = "population")

lines!(ax1, ts, [ξ.u(t)[1] for t in ts], linewidth = 2)

lines!(ax2, ts, [ξ.x(t)[1]^2+ξ.x(t)[7]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax2, ts, [ξ.x(t)[2]^2+ξ.x(t)[8]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax2, ts, [ξ.x(t)[3]^2+ξ.x(t)[9]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax2, position = :rc)

lines!(ax3, ts, [ξ.x(t)[4]^2+ξ.x(t)[10]^2 for t in ts], linewidth = 2, label = "|0⟩")
lines!(ax3, ts, [ξ.x(t)[5]^2+ξ.x(t)[11]^2 for t in ts], linewidth = 2, label = "|1⟩")
lines!(ax3, ts, [ξ.x(t)[6]^2+ξ.x(t)[12]^2 for t in ts], linewidth = 2, label = "|2⟩")
axislegend(ax3, position = :rc)

display(fig)
```
![image description](./Xgate.png)
## Lane Change
Coming soon!