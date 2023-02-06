# Cheat Sheet
Updated and adapted to match this implementation a bit more. Everything acts under a parameter set $θ$.

PRONTO iterates $ξ$, $φ$ is the reference trajectory

## Requirements
Dynamics: $ f(x(t),u(t),t,θ) $

Stage Cost $ l(x(t),u(t),t,θ) $

Terminal Cost $ p(x(t),u(t),t,θ) $

Regulator Matrices $ R(x(t),u(t),t,θ), Q(x(t),u(t),t,θ) $

## PRONTO

## Regulator
The regulator is computed by:

$$K_r(t) = R_r(t)^{-1} B_r(t)^T P_r(t)$$

Defined by evaluating along α(t),μ(t):

$$A_r(t) = f_{x}(α(t),μ(t),t,θ)$$
$$B_r(t) = f_{u}(α(t),μ(t),t,θ)$$
$$R_r(t) = R(α(t),μ(t),t,θ)$$
$$Q_r(t) = Q(α(t),μ(t),t,θ)$$

Where $P_r$ is found  by solving a differential riccati equation backwards in time from $P(α(T),μ(T),T)$.

$$-\dot{P}_r = A_r^T P_r + P_r A_r - K_r^T R_r K_r + Q_r$$




## Optimizer
$$K_o(t) = R_o^{-1}(S_o^T + B^T P)$$


## Second Order

$$Q_o = L_{xx} = l_{xx} + \sum λ_k f_{k,xx}$$