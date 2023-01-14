# PRONTO.jl

A julia implementation of the **PR**ojection-**O**perator-Based **N**ewton’s Method for **T**rajectory
**O**ptimization (PRONTO). PRONTO is a numerical method for trajectory optimization that performs computations directly in infinite-dimensional functional space, unlike most conventional methods, which transform the problem into a finite-dimensional approximation.


## Usage
This API is still very likely to change. Specifically regarding the symbolic generation of jacobians/hessians and passing them to the solver with the release of Julia 1.9.
