## --------------------------------- startup --------------------------------- ##
using Revise
using BenchmarkTools, JET
using PRONTO
using LinearAlgebra
using StaticArrays
using FastClosures

NX = 4
NU = 1
T = 10
ts = 0:0.001:T
model = (
    NX = NX,
    NU = NU,
    T = T, #TODO: remove
    ts = ts, #TODO: remove
    x0 = [0.0;1.0;0.0;0.0],
    u0 = zeros(NU),
    xf = [1.0;0.0;0.0;0.0], #For this example, we don't have any equilibrium points... xf is the target state here
    uf = zeros(NU),
    tol = 1e-5, #TODO: remove
    maxiters = 10, #TODO: remove
)

const α = Interpolant(t->PRONTO.guess(t, model.x0, model.xf, T), ts)
const μ = Interpolant(t->zeros(NU), ts)
const x = Interpolant(t->zeros(NX), ts)
const u = Interpolant(t->zeros(NU), ts)
const z = Interpolant(t->zeros(NX), ts)
const v = Interpolant(t->zeros(NU), ts)


## --------------------------------- model def & autodiff --------------------------------- ##

# model dynamics
H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
f = @closure (x,u) -> collect((H0 + u[1]*H1)*x)

# stage cost
Ql = zeros(NX,NX)
Rl = 0.01
Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1] #terminal cost matrix Pl

l = (x,u) -> 1/2*collect(x)'*Ql*collect(x) + 1/2*collect(u)'*Rl*collect(u)
p = (x) -> 1/2*collect(x)'*Pl*collect(x)

@info "running autodiff"
model = autodiff(model,f,l,p)
# model = PRONTO.Model(NX,NU,f,l,p) # returns a Model{NX,NU}
@info "autodiff complete"


model = merge(model, (
    Qr = let M = Diagonal(SMatrix{NX,NX}(diagm([1,1,1,1])))
        (t)->M
    end, # can Qr be a function of α?
    Rr = let M = Diagonal(SMatrix{NU,NU}(diagm([1])))
        (t)->M
    end,
    iRr = let M = Diagonal(SMatrix{NU,NU}(inv(diagm([1]))))
        (t)->M
    end
))

## ---------------------------------  --------------------------------- ##
A = functor(model.fx!,x,u,NX,NX)


fx! = model.fx!
_Ar = @buffer NX NX
Ar = @closure (t)->(fx!(_Ar,α(t),μ(t)); return _Ar)
_A = @buffer NX NX
A = @closure (x,u)->(fx!(_A,x,u); return _A)
# A = @functor (x,u)->(fx!(_A,x,u); return _A) NX NX

# function functor(f!,x,u,dims...)
#     A = buffer(dims...)
#     return @closure (t)->(f!(A,x(t),u(t)); return A)
# end


# A = make_A(model.fx!,x,u)

# macro buffer(S...)
#     return :(Buffer{Tuple{$(S...)}}())
# end


## --------------------------------- pronto components --------------------------------- ##
Kr = PRONTO.regulator(α,μ,model)
x! = projection_x(x0,α,μ,Kr,model)
update!(x, x!)
u! = projection_u(x,α,μ,Kr,model)
update!(u, u!)
Ko = optimizer(x,u,PT(α),model) 

## ---------------------------------  --------------------------------- ##


Kr = PRONTO.regulator(α,μ,model)
Kr(t)

@benchmark α(t) # 47 ns
@benchmark μ(t) # 47 ns

@benchmark Ar(t) # 92 ns
@benchmark fx!(_Ar, α(t), μ(t)) # 105 ns
@benchmark fx!(_Ar, $(α(t)), $(μ(t))) # 17 ns
# ie. most time is spent in interpolant lookup

@benchmark A($(α(t)), $(μ(t))) # 31 ns

@benchmark Kr(t)
@code_warntype Kr(t)
@report_opt Kr(t)
# 303 ns ... 755 ns
# 4 allocs, clean opt report

@benchmark Pr(t)
# 228 ns ... 731 ns
# 4 allocs, clean opt report
OrdinaryDiffEq.ODECompositeSolution