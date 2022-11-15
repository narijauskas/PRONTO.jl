# PRONTO Devnotes

## 1. Model Definition

The model requires:
$
f(θ,t,x,u)
$

```julia
using PRONTO

@model TwoSpin begin
    NX = 4; NU = 1; NΘ = 0

    f(θ,t,x,u) = ...
end
```

This defines a type:
```julia
TwoSpin <: PRONTO.Model{4,1,0}
```
## 2. Model Derivation
Symbolically generate functions for:

$
f(θ,t,ξ)
$

defined within the pronto module as:

```julia
PRONTO.f(M,θ,t,ξ)
PRONTO.f!(M,buf,θ,t,ξ)
```

dispatch on `M` where `M <: PRONTO.Model`

Macros do the heavy lifting.


Jacobian defined as:

$J_x (f) = \begin{bmatrix}
\frac{δf_?}{δx_?} & \cdots & \frac{δf_?}{δx_?}\\
\vdots & \ddots & \vdots\\
\frac{δf_?}{δx_?} & \cdots & \frac{δf_?}{δx_?}\\
\end{bmatrix}$


$f_x = J_x(f);\ f_{xu} = J_u(J_x(f))$

$f[2] = f_2$
$f_x[2,1] = \frac{δf_2}{δx_1}$
$f_{xu}[2,1,3] = \frac{δ\frac{δf_2}{δx_1}}{δu_3}$
$\vdots$

```julia
f[2] = f_2 # NX
fx[2,1] = df2/dx1 # NX,NX
fxu[2,1,3] = d(df2/dx1)/du3# NX,NX,NU
```

## 3. Intermediate Operators
Provide convenience for efficiently defining diffeq's later on. Eg.

$K_r = R_r^{-1} B_r^T P_r$

```julia
Ar(M,θ,t,φ) = (fx!(M,buf,θ,t,φ); return buf)
Kr(M,θ,t,φ,Pr) = Rr(...)\(Br(...)'Pr(...))
```

!!! warning "yo"
    - should these be in-place?
    - if so, where should buffers be defined?
    - this behavior would almost be closer to an ode solution

## 4. DiffEQs

$\frac{dPr}{dt} = -A_r^T P_r - P_r A_r + K_r^T R_r K_r - Q_r$

define an in-place version using the format of DifferentialEquations.jl

```julia
dPr_dt(dPr, Pr, (M,θ,φ), t)
```
## 5. ODE Solutions
uses internal buffer and is wrapped by FunctionWrappers.jl - todo: make thread-safe

```julia
Pr(t)
```

## 6. Trajectories
special ODE solutions with pretty plots :)

## 0. Buffered Operators
ODE solutions, trajectories, and intermediate operators all behave the same way:
- apply function in-place using internal buffer (with clear thread-safety)
- return a copy of the internal buffer (can use fancy types, eg. sparse arrays)

With some differences:
- ODEs/Trajectories render as plots (why not everything?)
- intermediate operators and diffeqs should both have symbolic rendering (maybe a macro facilitates this, eg. `@symbolic TwoSpin fx`)

