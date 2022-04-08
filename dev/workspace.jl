import Pkg; Pkg.activate(".")
using BenchmarkTools
using Symbolics
using PRONTO
using PRONTO: jacobian

##
@variables x[1:2] u[1:1]
# dynamics
g = 9.81
L = 2
f(x,u) = [x[2];   g/L*sin(x[1])-u[1]*cos(x[1])/L]

fx = jacobian(x, f, x, u)
fu = jacobian(u, f, x, u)

fxx = jacobian(x, fx, x, u)
fxu = jacobian(u, fx, x, u)
fuu = jacobian(u, fu, x, u)





##












using DataInterpolations
using Symbolics: derivative
using MatrixEquations

# @benchmark g = t->sin(t+mean(x))
# g = t->sin(t+mean(x))
# x = zeros(1000);

# g(1)

println("hello!")

# fx = jacobian(f_x)


# to compare against PRONTO for Dummies

# dynamics
g = 9.81
L = 2

# f(x,u,t) = f(x(t), u(t))
f(x,u) = [x[2];   g/L*sin(x[1])-u*cos(x[1])/L]
# f(x,u) = [x[1]; x[2]]
@variables x[1:2] u

function jacobian(dx, f, args...; inplace = false)
    f_sym = f(args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(1:length(f_sym)) do j
            derivative(f_sym[j], dx[i])
        end
    end...; dims = ndims(f_sym)+1)

    return eval(build_function(fx_sym, args...)[inplace ? 2 : 1])
end

fx = jacobian(x, f, x, u)
fu = jacobian(u, f, x, u)

fxx = jacobian(x, fx, x, u)
fxu = jacobian(u, fx, x, u)
fuu = jacobian(u, fu, x, u)





T = 10
t = 0:0.01:T # 99 ns
X = LinearInterpolation(zeros(2, length(t)), t) # 12 μs
U = LinearInterpolation(zeros(length(t)), t)

# non-allocating, lazily evaluated only at values needed by ode solver
A = t -> fx(X(t), U(t))
B = t -> fu(X(t), U(t))

# lqr stage cost
Qr = diagm([10, 1])
Rr = 1e-3

P_lqr,_ = arec(A(T), B(T)inv(Rr)B(T)', Qr) # solve algebraic riccati eq at time T

function riccati!(dP, P, (A,B,Q,R), t)
    Kr = inv(Rr)*B(t)'*P
    dP .= -A(t)'P - P*A(t) + Kr'*Rr*Kr - Q
end

P = solve(ODEProblem(riccati!, P_lqr, (T, 0.0), (A,B,Qr,Rr))) # solve differential riccati backwards in time

Kr = t->inv(Rr)*B(t)'*P(t)

