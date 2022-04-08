# to compare against PRONTO for Dummies

# dynamics
g = 9.81;
L = 2;

f(x,u,t) = f(x(t), u(t))
f(x,u) = [x[2];   g/L*sin(x[1])-u*cos(x[1])/L]

# incremental cost
Q = eye(2);
R = 1;

l(x,u,t) = l(x(t), u(t))
l(x,u) = 1/2*x'*Q*x + 1/2*R*u^2


# states/allocations
# x: [2,1]xT
# u: 1xT


A = t->fx(x(t),u(t))