# Dev Notes and Version History

## TODO:

### General
  - website infrastructure
  - expand documentation
    - detailed examples
    - problem setup requirements & syntax
    - what goes on under the hood
  - cleanup devnotes
  - consolidate/cleanup tests/examples
    - get rid of octopus
    - clean up lambda

### Code functionality
  - store intermediate trajectories/matrices
  - generate Dh plot
  - debug options
  - vectorized parameters

## v0.4.0
  - refactor v0.3 with fewer symbolic intermediates (might change with Julia 1.9)
  - interpolated inputs for trajectories





## v0.4.0-dev
  - refactor v0.3 with optional symbolics
  - generic methods for all steps -> dispatch on M
  - sparsity-aware, consolidated symbolics?
  - build chains with LinearOperators.jl
  - Triangular dPr structure (ie, make dPr a vector)
  - @model macro to provide isolated eval of user code



## v0.3.0
    - unreleased
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

## v0.5.x
  - stabilize API