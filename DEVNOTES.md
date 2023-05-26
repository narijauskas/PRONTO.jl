# Dev Notes and Version History

## Code Gen
  - unit testing codegen #DONE: finish TwoSpin, verify
  - support partial model regeneration #DONE:
  - fix needing to collect(x') #DONE:
  - codegen interface rework #DONE:
  - codegen rewrite #DONE:
  - support for vector/matrix parameters #DONE:

## Package
  - remove unneeded dependencies #DONE:
  - set compats #YO: mostly done, set julia, update symbolics via breaking change
  - switch DifferentialEquations->OrdinaryDiffEQ #TODO:
  - clean old files #TODO:

## Website
  - fix versioning #TODO:
  - fix badge #TODO:
  - process PRs #YO: do these direct to main
  - update docs

## Solver Core
  - solver options interface #FUTURE:
  - save intermediate steps #FUTURE:
  - Dh plots #FUTURE:
  - finalize model type #DONE:
  - vectorized parameters #DONE:

## Don't Forget
  - LinearOperators
  - TimerOutputs


### General
  - website infrastructure
  - expand documentation
    - detailed examples
    - problem setup requirements & syntax
    - what goes on under the hood
    - function wrappers, etc.
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