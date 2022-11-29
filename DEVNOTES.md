# Dev Notes and Version History

## v0.4.0-dev
  - refactor v0.3 with optional symbolics
  - generic methods for all steps -> dispatch on M
  - sparsity-aware, consolidated symbolics?
  - build chains with LinearOperators.jl
  - Triangular dPr structure (ie, make dPr a vector)
  - @model macro to provide isolated eval of user code


## v0.3.0-dev
    - rewrite with new dispatch-on-model function pasing structure
    - newton step
    - inline plots
    - parametric models
    - symbolic previews of intermediate equations
    - no runtime compile
    - faster precompile
    - variable timestep ODE solutions in FunctionWrapper

## v0.2.0
    - code refactor
    - custom linear interpolant
    - buffered closures (capture autodiff)
    - type stable buffer


## v0.1.0
    - package setup
    - working autodiff
    - working algorithm (gradient descent)
    - stable base API
    - serves as a reference point for future improvement/optimization


--- TODO: ---

## v0.3.0
    - Dh plots
    - plot grid layout
    - verbosity levels
    - pass parameters
    - jacobian rework for new macros
    - @model pass by expression?
    - consolidate test sets

## v0.x.x
    - write benchmark sets
    - write example problems
    - write test sets
    - write documentation
