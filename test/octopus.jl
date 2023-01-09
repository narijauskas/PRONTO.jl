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


# fg = (ρ_arm - ρ_sea)*V_arm*g
# V_arm is a 4Nx4N matrix, and g must be vectorized somehow
# V is constant due to hydrostat constraint


# ----------------------------------- fw: drag forces ----------------------------------- #

c_per = 1.013 # dimensionless perpendicular drag (experimental)
c_tan = 0.0256 # dimensionless tangential drag (experimental)
#TODO: convert per/tan to x/y at each mass - transformation matrix?

# ----------------------------------- fc: hydrostat constraint ----------------------------------- #

fc(x) = C*p(x)
p(x) = inv(G*inv(M)*C)*(γ-G*inv(M)*(fm(x) + fg(x) + fw(x)))

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
x_v(i) = q[1+4(i-1)]
y_v(i) = q[2+4(i-1)]
x_d(i) = q[3+4(i-1)]
y_d(i) = q[4+4(i-1)]
# dq is q[4n+1:8n]
dq(i) = q[i+4n]
dx_v(i) = q[1+4(i-1)+4n]
dy_v(i) = q[2+4(i-1)+4n]
dx_d(i) = q[3+4(i-1)+4n]
dy_d(i) = q[4+4(i-1)+4n]

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




## ----------------------------------- relaxed geometry ----------------------------------- ##
wr = LinRange(0.015, 0.001, n) # height/width of each segment boundary
lr = 0.240/(n-1) # spacing of segment boundaries

# constant segment area - equal to D*q_rest
sc = [lr*(wr[i]+wr[i+1])/2 for i in 1:n-1]

# constant segment volume
V = [lr/3*(wr[i]^2 + wr[i+1]^2 + wr[i]*wr[i+1]) for i in 1:n-1]

# mass matrix - each of the 2n masses is repeated twice (4nx4n):
M = diagm(ρ_arm/4 * repeat([V[1]; [V[i-1]+V[i] for i in 2:n-1]; V[n-1]], inner=4))

# relaxed positions and velocities
q_relaxed = [reduce(vcat, [lr*(i-1); -wr[i]/2; lr*(i-1); wr[i]/2] for i in 1:n); zeros(4n)]

# relaxed compartment edge width (n)
# wr = d_seg
# relaxed compartment edge length (n-1)
# lr = [sqrt(l_seg^2 + 1/4*(d_seg[i+1]-d_seg[i])^2) for i in 1:n-1] # (~l_seg)







# ----------------------------------- fg: gravity/buoyancy ----------------------------------- #

ρ_arm = 1042 # kg/m^3, arm density
ρ_sea = 1022 # kg/m^3, seawater density

# buoyancy matrix - same form as mass matrix (4nx4n)
B = diagm((ρ_arm - ρ_sea)/4 * repeat([V[1]; [V[i-1]+V[i] for i in 2:n-1]; V[n-1]], inner=4))

# gravity
g = repeat([0, -9.81], outer=2n)

# constant
fg = B*g


## ----------------------------------- fm: muscle forces ----------------------------------- ##
T_passive = 2.0e4 # N/m^2
T_active = 1.32e5 # N/m^2
α0 = 9 # Ns/m^2


# create j muscles: n transverse, n-1 ventral, n-1 dorsal
# the ith compartment is surrounded by:
    # transverse ith and (i+1)th muscles
    # ventral (i+n)th muscle
    # dorsal (i+2n-1)th muscle

# rest length of linear spring
# l0 = 0.4*lm; where lm = 1.25*l_relaxed
lo_t(i) = 0.5*wr[i]
lo_v(i) = 0.5*lr
lo_d(i) = 0.5*lr


# muscle "action moment" - relaxed cross sectional area divided by relaxed length
z_t(i) = lr
z_d(i) = ((wr[i]+wr[i+1])/2)^2 / (2*lr)
z_v(i) = ((wr[i]+wr[i+1])/2)^2 / (2*lr)

# n transverse muscle vectors point from ventral to dorsal mass
μ_t(i) = [x_d(i) - x_v(i), y_d(i) - y_v(i)]
dμ_t(i) = [dx_d(i) - dx_v(i), dy_d(i) - dy_v(i)]
l_t(i) = norm(μ_t(i))
dl_t(i) = norm(dμ_t(i))
e_t(i) = μ_t(i)/l_t(i)
de_t(i) = μ_t(i)/dl_t(i)

# n-1 ventral muscle vectors point from ith to (i+1)th ventral mass
μ_v(i) = [x_v(i+1) - x_v(i), y_v(i+1) - y_v(i)]
dμ_v(i) = [dx_v(i+1) - dx_v(i), dy_v(i+1) - dy_v(i)]
l_v(i) = norm(μ_v(i))
dl_v(i) = norm(dμ_v(i))
e_v(i) = μ_v(i)/l_v(i)
de_v(i) = μ_v(i)/dl_v(i)

# n-1 dorsal muscle vectors point from ith to (i+1)th dorsal mass
μ_d(i) = [x_d(i+1) - x_d(i), y_d(i+1) - y_d(i)]
dμ_d(i) = [dx_d(i+1) - dx_d(i), dy_d(i+1) - dy_d(i)]
l_d(i) = norm(μ_d(i))
dl_d(i) = norm(dμ_d(i))
e_d(i) = μ_d(i)/l_d(i)
de_d(i) = μ_d(i)/dl_d(i)


# muscle activation input: n transverse, n-1 ventral, n-1 dorsal
@variables u[1:(3n-2)]
u_t(i) = u[i]
u_v(i) = u[i+n]
u_d(i) = u[i+2n-1]



f_t(i) = z_t(i)*((T_passive + T_active*u_t(i))*(l_t(i) - lo_t(i)) + α0*dl_t(i))
f_v(i) = z_v(i)*((T_passive + T_active*u_v(i))*(l_v(i) - lo_v(i)) + α0*dl_v(i))
f_d(i) = z_d(i)*((T_passive + T_active*u_d(i))*(l_d(i) - lo_d(i)) + α0*dl_d(i))

# net force on ith ventral mass (2x1)
function F_v(i)
    i == 1 ? f_t(1)*e_t(1) + f_v(1)*e_v(1) :
    i == n ? f_t(n)*e_t(n) - f_v(n-1)*e_v(n-1) :
    f_t(i)*e_t(i) + f_v(i)*e_v(i) - f_v(i-1)*e_v(i-1)
end

# net force on ith dorsal mass (2x1)
function F_d(i)
    i == 1 ? - f_t(1)*e_t(1) + f_d(1)*e_d(1) :
    i == n ? - f_t(n)*e_t(n) - f_d(n-1)*e_d(n-1) :
    - f_t(i)*e_t(i) + f_d(i)*e_d(i) - f_d(i-1)*e_d(i-1)
end

