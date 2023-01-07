#=
implementation of PRONTO for the model presented in Yekutieli et al 2005
Dynamic Model of the Octopus Arm. I. Biomechanics of the Octopus Reaching Movement
=#

# parameters
# Arm width is 15 mm at thebase and 1 mm at the tip.
# Arm length is 240 mm and the depthat every point equals the width at that point and remainsconstant throughout the simulation



# states:
# q = ...
# size 4n

# inputs: u = a(t) ∈ [0,1]
# size 3n-2 or n-1 depending on assumptions

# model dynamics
q̈ = inv(M)*(fm(x,u)+fg(x)+fw(x)+fc(x))



# ----------------------------------- fg: gravity/buoyancy ----------------------------------- #

ρ_arm = 1042 # kg/m^3, arm density
ρ_sea = 1022 # kg/m^3, seawater density
fg = (ρ_arm - ρ_sea)*V_arm*g
# V_arm is a 4Nx4N matrix, and g must be vectorized somehow
# V is constant due to hydrostat constraint


# ----------------------------------- fw: drag forces ----------------------------------- #

c_per = 1.013 # dimensionless perpendicular drag (experimental)
c_tan = 0.0256 # dimensionless tangential drag (experimental)

# ----------------------------------- fc: hydrostat constraint ----------------------------------- #

fc(x) = C*p(x)
p(x) = inv(G*inv(M)*C)*(γ-G*inv(M)*(fm(x) + fg(x) + fw(x)))
# gamma?

# ----------------------------------- fm: muscle forces ----------------------------------- #
# fm: muscle forces
fm(x,u)

# nonlinear muscle: use linear, should be close
fm(x,u) = A*(a(t)*Fa*(l(t)/lm)*Fv(v/vmax) + Fp*(l(t)/lm))
l(t) # muscle length
lm # specific muscle length (length at peak force)
A_long = wr^2 / 2
A_tran = lr*wr
# note d = wr
# wr is relaxed width/depth
# lr is relaxed length
# lr ≈ 08lm

# Fa and Fp are data driven normalized curves... 🤮
# scaled by maximal stress: 8e4 N/m^2

#Fv is piecewise... use atan smooth approx or see Zajac 1989
# Fv(v⋆) = 
# 0 if v⋆ ≤ -1
# 1+v⋆ if -1 < v⋆ < 0.8
# 1.8 if 0.8 ≥ v⋆
# where v⋆ = v/vmax
# vmax = 10*lm / sec (10 muscle lengths/sec)





# ----------------------------------- cost functions, etc. ----------------------------------- #

# cost functions for reaching
# stage cost: ??
# terminal cost: extended arm?

# regultor settings
# - equal priority
# - prioritize medial muscles
# - prioritize lateral muscles

# comparison of optimal trajectory to open-loop propogating wave input
# - generate neural activation pattern u_neural(t)


##
using Symbolics, LinearAlgebra
n = 4 # 2 segments
@variables q[1:8n]
# q is q[1:4n]
# dq is q[4n+1:8n]
x_v(q,i) = q[1+4(i-1)]
y_v(q,i) = q[2+4(i-1)]
x_d(q,i) = q[3+4(i-1)]
y_d(q,i) = q[4+4(i-1)]
# closure version:
x_v(i) = x_v(q,i)
y_v(i) = y_v(q,i)
x_d(i) = x_d(q,i)
y_d(i) = y_d(q,i)
dq(q,j) = q[4n+j]

# C matrix
# λ_i-1 column
Λ(i) = [
    y_d(i) - y_v(i-1);
    x_v(i-1) - x_d(i);
    y_d(i-1) - y_v(i);
    x_v(i) - x_d(i-1);
]

# λ_i column
λ(i) = [
    y_v(i+1) - y_d(i);
    x_d(i) - x_v(i+1);
    y_v(i) - y_d(i+1);
    x_d(i+1) - x_v(i);
]

Cblock(i) = [zeros(4,i-2);; Λ(i);; λ(i);; zeros(4,n-1-i)]
C = 1/2 .* [[λ(1);; zeros(4,n-2)]; [Cblock(i) for i in 2:n-1]...; [zeros(4,n-2);; Λ(n)]]

# D:q->s
# row block starting at index i for construction of D matrix
function Dblock(i)
    j = 4*(i-1)
    qx = [4,3,8,7,2,1,6,5] # result of shoelace formula
    1/2 .* [zeros(j); [iseven(k) ? -q[j+k] : q[j+k] for k in qx]; zeros(4n-8-j)]
end
D = reduce(vcat, Dblock(i)' for i in 1:n-1)


δD_δq(D_ik,q_k) = isequal(D_ik, q_k) ? 1 : isequal(-D_ik, q_k) ? -1 : 0
G = [sum([D[i,k] + δD_δq(D[i,k],q[k])*q[l] for k in 1:4n]) for i in 1:n-1, l in 1:4n]
γ = [-2*sum(δD_δq(D[i,k],q[l])*dq(l)*q[k] for l in 1:4n, k in 1:4n) for i in 1:n-1]





@variables m[1:4n]
M = diagm(collect(m))

Minv = inv(M'M)*M

G*inv(M)

# ----------------------------------- rest length ----------------------------------- #
d = LinRange(0.015, 0.001, n) # height/width of each segment boundary
l = 0.240/(n-1) # spacing of segment boundaries
q_rest = [reduce(vcat, [l*(i-1); -d[i]/2; l*(i-1); d[i]/2] for i in 1:n); zeros(4n)]
# [x_v; y_v; x_d; y_d]
# segment volume (constant!)
V = [l/3*(d[i]^2 + d[i+1]^2 + d[i]*d[i+1]) for i in 1:n-1]

# masses:
(ρ_arm*V[1])/4
(ρ_arm*(V[i-1]+V[i]))/4
(ρ_arm*V[n-1])/4
# reduce(vcat, [kron([1;1], i) for i in 1:3])

S(q) = -q[4]*q[1] +q[3]*q[2] -q[8]*q[3] +q[7]*q[4] -q[2]*q[5] +q[1]*q[6] +q[5]*q[8] -q[6]*q[7]
T(q) = q[1]*(q[6]-q[4]) - q[5]*(q[2]-q[4]) + q[3]*(q[2]-q[6])

sc = [l*(d[i]+d[i+1])/2 for i in 1:n-1]