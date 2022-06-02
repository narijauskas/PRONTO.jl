
## Problem

The ODE solution type from SciMLBase is very complex in terms of parametric type signature:

```julia
struct ODESolution{T,N,uType,uType2,DType,tType,rateType,P,A,IType,DE} <: AbstractODESolution{T,N,uType}
  u::uType
  u_analytic::uType2
  errors::DType
  t::tType
  k::rateType
  prob::P
  alg::A
  interp::IType
  dense::Bool
  tslocation::Int
  destats::DE
  retcode::Symbol
end
```

If used as a parameter in another ODEProblem, the compile-time complexity will grow rapidly.

## However
We can't just easily copy the interpolant, because the interpolant is designed to be flexible, and therefore *generic*. IType can often be just as bad in terms of complexity.

## Solution
Make a SolutionWrapper type that wraps IType. Has simple type signature. (at most parametrize return type of X)

Write getter functions X(t), which know return type
Possibly `@nospecialize` on getter.

Luckily, it seems interp is callable :)


So,
1. make a SolutionWrapper type which holds interp data
2. make a constructor which takes generic ODESolutions
3. make it callable, forwarding to the internal solution type
4. nospecialize if needed