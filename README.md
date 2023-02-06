
<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://narijauskas.github.io/PRONTO.jl/stable) -->
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://narijauskas.github.io/PRONTO.jl/dev)

# PRONTO.jl
A Julia implementation of the **PR**ojection-**O**perator-Based **N**ewton’s Method for **T**rajectory **O**ptimization (PRONTO). PRONTO is a numerical method for trajectory optimization which leverages variational calculus to solve the optimal control problem directly in infinite-dimensional function space. Consider the optimal control problem:

$$\min h(ξ) = p(x(T)) + \int_0^T l(x(t),u(t)) dt$$
where $h$ is $\mathcal{C}^2$ convex function, and $ξ(t) = (x(t),u(t))$ is a trajectory restricted by the dynamics of the system, which lies on the trajectory manifold:

$$\mathcal{T} = \{ ξ ∈(\mathcal{X}\times\mathcal{U})\ |\ \dot{x} = f(x,u),\ x(0)=x_0 \}$$

To explore beyond this trajectory manifold, we consider an arbitrary curve $η ∈(\mathcal{X}\times\mathcal{U})$ where $η(t) = (α(t),μ(t))$. The projection operator $ \mathcal{P}:(\mathcal{X}\times\mathcal{U})\rightarrow \mathcal{T}$, which maps curves onto the trajectory manifold of the system, is used to convert the above constrained optimization problem to the unconstrained problem:

$$g(η) = h(\mathcal{P}(η))$$

Given the current trajectory iterate $ξ_k$, we wish to compute the descent direction $ζ_k$, which lies in the tangent space of $\mathcal{T}$. This is done by solving the optimization problem:

$$ζ_k = \arg\min Dg(ξ_k)\cdotζ + \frac{1}{2}D^2g(ξ_k)\cdot(ζ,ζ)$$

where $Dg$ and $D^2g$ are the first and second Fréchet derivatives of $g$.

This optimization problem is first converted into a linear-quadratic optimal control problem, and then solved in one of two ways. First, we try to solve a Newton descent direction, which does not have a guaranteed solution, but provides quadratic convergence. If this fails, we switch to a quasi-Newton descent direction, which has a guaranteed solution and provides super-linear convergence.

Because $ζ_k$ is computed by local second order approximation, there is no guarantee that $g(ξ_k + ζ_k) < g(ξ_k)$. Consequently we find a step size $γ_k ≤ 1$ via backtracking line search which enforces the Armijo rule.

After calculating a descent direction $ζ_k$ and a step size $γ_k$ the projection operator is used to ensure the next solution iterate lies on the trajectory manifold.

$$ ξ_{k+1} = \mathcal{P}(ξ_k + γ_k ζ_k)$$

For the mathematical details of the PRONTO algorithm, please refer to: [(Hauser 2002)](https://www.sciencedirect.com/science/article/pii/S1474667015387334), [(Hauser 2003)](https://ieeexplore.ieee.org/abstract/document/1243395), and [(Shao et al. 2022)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.032605)

## Usage
Please see the examples folder for usage examples. This API is still very likely to change – especially regarding the symbolic generation of jacobians/hessians and passing them to the solver. Note that upcoming changes in Julia 1.9 should substantially improve the compile time of large generated functions ([#45276](https://github.com/JuliaLang/julia/issues/45276), [#45404](https://github.com/JuliaLang/julia/issues/45404)).

To see the generated definition of any model function, run, eg. `methods(PRONTO.f)` and open the temporary file where the definition is stored.

## Upcoming Changes
- Fine grained control of verbosity/runtime-feedback.
- Support for SVector parameters (eg. regulator Q/R matrix diagonals).
- Support for constraints [(Dearing et al. 2022)](https://arc.aiaa.org/doi/full/10.2514/1.G006166), [(Hauser & Saccon 2006)](https://ieeexplore.ieee.org/abstract/document/4178067)
- Easy access to differential equation solver options (algorithm, tolerance, etc.).
- More options for guess trajectories (eg. smooth atan)
