using PRONTO

@model TwoSpinP begin
    NX = 4; NU = 1; NΘ = 1

    # model dynamics
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    f(θ,t,x,u) = (H0 + u[1]*H1)*x

    # stage cost
    Ql = zeros(NX,NX)
    Rl = [0.01]
    l(θ,t,x,u) = 1/2*x'*Ql*x .+ 1/2*u'*Rl*u

    # terminal cost
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    p(θ,t,x,u) = 1/2*x'*Pl*x
end
#

M = TwoSpinP()
θ = [1.0]
t0 = 0.0
tf = 10.0
x0 = [0.0;1.0;0.0;0.0]
xf = [1.0;0.0;0.0;0.0]
u0 = [0.0]
ug = @closure t -> u0

φg = PRONTO.guess_ol(M,θ,t0,tf,x0,ug)

mod = PRONTO.@test1 begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    f = (θ,t,x,u) -> collect((H0 + u[1]*H1)*x)
end

@variables θ[1:NΘ]
@variables t
@variables x[1:NX] 
@variables u[1:NU]

mod.f(θ,t,x,u)
f_sym = collect(Base.invokelatest(mod.f,θ,t,x,u))
f_ex,f!_ex = build_function(f_sym,θ,t,x,u)
# defines an anonymous function, eg. 


M = PRONTO.test5(:Yeet,
    quote
        NX = 4; NU = 1; NΘ = 1
        # model dynamics
        H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
        H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
        f(θ,t,x,u) = (H0 + u[1]*H1)*x
    end
)

write("temp.jl", prod(string.(M).*"\n"))

ex = PRONTO.cleanup(f!_ex)

function test3(ex)
    mod = eval(:(module $(gensym()) $ex end))
    println("created a temporary module at $mod")
    mod.NX
end




ex1 = (quote
    function (x,y)
    x + y
    end
end).args[2]


ex2 = (quote
    function bar(x,y)
    x + y
    end
end).args[2]
# goal: map from user defined f_user to processed f_ex and f!_ex

ex3 = (quote
bar(x,y) = x+y
end).args[2]


ex1 == ex2 # false
ex1.args[1] = :(bar(x,y))
ex1 == ex2 # true

# or...
ex1.args[1] = :(PRONTO.f(M::$M,θ,t,ξ))

# make function to replace
:((x,y))
# with
:(bar(x,y))
# and ex1 can be turned into ex2!

PRONTO.@snoop

function bar()
    local x=0
    eval(:(x=2))
    return x
end

using SafeTestsets
@safetestset



bar(x,y) = x+y

module Foo
    foo(x,y) = bar(x,y)
end





using Symbolics
@variables x y

using SparseArrays
using LinearAlgebra
N = 8
A = sparse(Tridiagonal([x^i for i in 1:N-1], [x^i * y^(8-i) for i in 1:N], [y^i for i in 1:N-1]))
build_function(A,[x,y])