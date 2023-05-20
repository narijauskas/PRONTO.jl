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
# q̈ = inv(M)*(fm(x,u)+fg(x)+fw(x)+fc(x))



# ----------------------------------- fg: gravity/buoyancy ----------------------------------- #


# fg = (ρ_arm - ρ_sea)*V_arm*g
# V_arm is a 4Nx4N matrix, and g must be vectorized somehow
# V is constant due to hydrostat constraint


# ----------------------------------- fw: drag forces ----------------------------------- #


#TODO: convert per/tan to x/y at each mass - transformation matrix?



# ----------------------------------- fm: muscle forces ----------------------------------- #
# fm: muscle forces
# fm(x,u)

# # nonlinear muscle: use linear, should be close
# fm(x,u) = A*(a(t)*Fa*(l(t)/lm)*Fv(v/vmax) + Fp*(l(t)/lm))
# l(t) # muscle length
# lm # specific muscle length (length at peak force)
# A_long = wr^2 / 2
# A_tran = lr*wr
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

import Pkg; Pkg.activate()
## ----------------------------------- symbolics ----------------------------------- ##
using Symbolics, LinearAlgebra
n = 10 # number of segments

# states: 2n x/y positions and 2n x/y velocities
# format: [x_v; y_v; x_d; y_d]
@variables q[1:8n]
# positions
x_v(i) = q[1+4(i-1)]
y_v(i) = q[2+4(i-1)]
x_d(i) = q[3+4(i-1)]
y_d(i) = q[4+4(i-1)]
# velocities
dq(i) = q[i+4n]
dx_v(i) = q[1+4(i-1)+4n]
dy_v(i) = q[2+4(i-1)+4n]
dx_d(i) = q[3+4(i-1)+4n]
dy_d(i) = q[4+4(i-1)+4n]


# inputs: muscle activations: n transverse, n-1 ventral, n-1 dorsal
@variables u[1:(3n-2)]
u_t(i) = u[i]
u_v(i) = u[i+n]
u_d(i) = u[i+2n-1]



# ----------------------------------- relaxed geometry ----------------------------------- ##
wr = LinRange(0.015, 0.001, n) # relaxed height/width of each segment boundary
lr = 0.240/(n-1) # relaxed spacing of segment boundaries

# constant segment area - equal to D*q0
sc = [lr*(wr[i]+wr[i+1])/2 for i in 1:n-1]

# constant volume of n-1 segments
Vc = [1/3*lr*(wr[i]^2 + wr[i+1]^2 + wr[i]*wr[i+1]) for i in 1:n-1]


# relaxed positions and velocities
q0 = [reduce(vcat, [lr*(i-1); -wr[i]/2; lr*(i-1); wr[i]/2] for i in 1:n); zeros(4n)]
u0 = zeros(3n-2)


# relaxed compartment edge width (n)
# wr = d_seg
# relaxed compartment edge length (n-1)
# lr = [sqrt(l_seg^2 + 1/4*(d_seg[i+1]-d_seg[i])^2) for i in 1:n-1] # (~l_seg)



# ----------------------------------- fg: gravity/buoyancy ----------------------------------- ##

ρ_arm = 1042 # kg/m^3, arm density
ρ_sea = 1022 # kg/m^3, seawater density

# mass matrix - each of the 2n masses is repeated twice (4nx4n):
M = diagm(ρ_arm/4 * repeat([Vc[1]; [Vc[i-1]+Vc[i] for i in 2:n-1]; Vc[n-1]], inner=4))

# buoyancy matrix - same form as mass matrix (4nx4n)
B = diagm((ρ_arm - ρ_sea)/4 * repeat([Vc[1]; [Vc[i-1]+Vc[i] for i in 2:n-1]; Vc[n-1]], inner=4))

# gravity
g = repeat([0, -9.81], outer=2n)

# constant
# fg = B*g
fg = B*g
# fg_fxn = eval(build_function(fg_sym)[1])


# ----------------------------------- fm: muscle forces ----------------------------------- ##
T_passive = 2.0e3 # N/m^2
T_active = 1.32e5 # N/m^2
α0 = 9 # Ns/m^2



# the ith compartment is surrounded by:
    # transverse ith and (i+1)th muscles
    # ventral (i+n)th muscle
    # dorsal (i+2n-1)th muscle

# rest length of linear spring: length for which the muscle produces zero force
# lo = 0.4*lm; where lm = 1.25*l_relaxed
lo_t(i) = 0.8*wr[i]
lo_v(i) = 0.8*lr
lo_d(i) = 0.8*lr


# muscle "action moment" - relaxed cross sectional area divided by relaxed length
z_t(i) = lr
z_d(i) = ((wr[i]+wr[i+1])/2)^2 / (2*lr)
z_v(i) = ((wr[i]+wr[i+1])/2)^2 / (2*lr)


# n transverse muscle vectors point from ventral to dorsal mass
μ_t(i) = [x_d(i) - x_v(i), y_d(i) - y_v(i)]
l_t(i) = norm(μ_t(i))
e_t(i) = μ_t(i)/l_t(i)

dμ_t(i) = [dx_d(i) - dx_v(i), dy_d(i) - dy_v(i)]
dl_t(i) = norm(dμ_t(i))
de_t(i) = μ_t(i)/dl_t(i)

f_t(i) = z_t(i)*((T_passive + T_active*u_t(i))*(l_t(i) - lo_t(i)) + α0*dl_t(i))


# n-1 ventral muscle vectors point from ith to (i+1)th ventral mass
μ_v(i) = [x_v(i+1) - x_v(i), y_v(i+1) - y_v(i)]
l_v(i) = norm(μ_v(i))
e_v(i) = μ_v(i)/l_v(i)

dμ_v(i) = [dx_v(i+1) - dx_v(i), dy_v(i+1) - dy_v(i)]
dl_v(i) = norm(dμ_v(i))
de_v(i) = μ_v(i)/dl_v(i)

f_v(i) = z_v(i)*((T_passive + T_active*u_v(i))*(l_v(i) - lo_v(i)) + α0*dl_v(i))


# n-1 dorsal muscle vectors point from ith to (i+1)th dorsal mass
μ_d(i) = [x_d(i+1) - x_d(i), y_d(i+1) - y_d(i)]
l_d(i) = norm(μ_d(i))
e_d(i) = μ_d(i)/l_d(i)

dμ_d(i) = [dx_d(i+1) - dx_d(i), dy_d(i+1) - dy_d(i)]
dl_d(i) = norm(dμ_d(i))
de_d(i) = μ_d(i)/dl_d(i)

f_d(i) = z_d(i)*((T_passive + T_active*u_d(i))*(l_d(i) - lo_d(i)) + α0*dl_d(i))


# for debugging:
Lt = eval(build_function([l_t(i) for i in 1:n], q)[1])
Lv = eval(build_function([l_v(i) for i in 1:n-1], q)[1])
Ld = eval(build_function([l_d(i) for i in 1:n-1], q)[1])

Ft = eval(build_function([f_t(i) for i in 1:n], q, u)[1])
Fv = eval(build_function([f_v(i) for i in 1:n-1], q, u)[1])
Fd = eval(build_function([f_d(i) for i in 1:n-1], q, u)[1])

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

fm_sym = reduce(vcat, [F_v(i); F_d(i)] for i in 1:n)
fm = eval(build_function(fm_sym, q, u)[1])



# ----------------------------------- fw: drag forces ----------------------------------- #


c_p = 1.013 # dimensionless perpendicular drag coefficient (experimental)
c_a = 0.0256 # dimensionless axial/tangential drag coefficient (experimental)

# D:q->s maps states to segment areas (n-1 x 4n)
# row block for construction of symbolic D matrix
function Dblock(i)
    j = 4*(i-1)
    qx = [4,3,8,7,2,1,6,5] # result of shoelace formula
    1/2 .* [zeros(j); [iseven(k) ? -q[j+k] : q[j+k] for k in qx]; zeros(4n-8-j)]
    # qx = [6,5,2,1,8,7,4,3]
    # 1/2 .* [zeros(j); [isodd(k) ? -q[j+k] : q[j+k] for k in qx]; zeros(4n-8-j)]
end
D = reduce(vcat, Dblock(i)' for i in 1:n-1)

# area of ith segment
s(i) = sum(D[i,k]*q[k] for k in 1:4n)



# tangential axis of the ith segment
μ_a(i) = 1/2 * (μ_v(i) + μ_d(i))
l_a(i) = norm(μ_a(i))
e_a(i) = μ_a(i)/l_a(i)

# perpendicular axis of the ith segment
μ_p(i) = [-μ_a(i)[2], μ_a(i)[1]]
e_p(i) = μ_p(i)/norm(μ_p(i))

# ith segment centroid velocity and components
v(i) = 1/4 * [dx_v(i)+dx_d(i)+dx_v(i+1)+dx_d(i+1); dy_v(i)+dy_d(i)+dy_v(i+1)+dy_d(i+1)]
v_a(i) = v(i) ⋅ e_a(i)
v_p(i) = v(i) ⋅ e_p(i)

# projected area (this could be more complex, and depend on direction of motion)
A_p(i) = 1/2*l_a(i)*(l_t(i)+l_t(i+1))

# segment surface area
A_s(i) = 2*s(i) + 1/2*(l_d(i) + l_v(i))*(l_t(i) + l_t(i+1))

# magnitude of axial (tangential) drag
f_a(i) = 1/2*ρ_sea*A_s(i)*c_a*norm(v_a(i))

# magnitude of perpendicular drag
f_p(i) = 1/2*ρ_sea*A_p(i)*c_p*norm(v_p(i))

# total drag vector for the ith dorsal or ith ventral mass
function F_w(i)
    i == 1 ? 1/4*(f_a(1)*e_a(1) + f_p(1)*e_p(1)) :
    i == n ? 1/4*(f_a(n-1)*e_a(n-1) + f_p(n-1)*e_p(n-1)) :
    1/4*(f_a(i)*e_a(i) + f_a(i-1)*e_a(i-1) + f_p(i)*e_p(i) + f_p(i-1)*e_p(i-1))
end

fw_sym = reduce(vcat, [F_w(i); F_w(i)] for i in 1:n)
fw = eval(build_function(fw_sym, q)[1])


# ----------------------------------- fc: hydrostat constraint ----------------------------------- ##

# C matrix
# λ_i-1 column for construction of symbolic C matrix
Λ(i) = [
    y_d(i) - y_v(i-1);
    x_v(i-1) - x_d(i);
    y_d(i-1) - y_v(i);
    x_v(i) - x_d(i-1);
]

# λ_i column for construction of symbolic C matrix
λ(i) = [
    y_v(i+1) - y_d(i);
    x_d(i) - x_v(i+1);
    y_v(i) - y_d(i+1);
    x_d(i+1) - x_v(i);
]

# row blocks for construction of symbolic C matrix
function Cblock(i)
    i == 1 ? [λ(1);; zeros(4,n-2)] :
    i == n ? [zeros(4,n-2);; Λ(n)] :
    [zeros(4,i-2);; Λ(i);; λ(i);; zeros(4,n-1-i)]
end
C_sym = 1/2 .* reduce(vcat, Cblock(i) for i in 1:n)
C0 = eval(build_function(C_sym, q)[1])
C = (q)->(reshape(C0(q), (4n,n-1)))


# symbolic G and γ
using Symbolics: derivative
G_sym = [sum([D[i,k] + derivative(D[i,k],q[k])*q[l] for k in 1:4n]) for i in 1:n-1, l in 1:4n]
G0 = eval(build_function(G_sym, q)[1])
G = (q)->(reshape(G0(q), (n-1, 4n)))


γ_sym = [-2*sum(derivative(D[i,k],q[l])*dq(l)*q[k] for l in 1:4n, k in 1:4n) for i in 1:n-1]
γ = eval(build_function(γ_sym, q)[1])


fc(q,u) = C(q)*pinv(G(q)*inv(M)*C(q))*(γ(q) - G(q)*inv(M)*(fm(q,u) + fw(q) + fg))
P(q,u) = pinv(G(q)*inv(M)*C(q))*(γ(q) - G(q)*inv(M)*(fm(q,u) + fw(q) + fg))


## ----------------------------------- dynamics ----------------------------------- ##

using OrdinaryDiffEq
using OrdinaryDiffEq: Tsit5, Rosenbrock23
# using UnicodePlots


function octopus(dq, q, u, t)
    # velocity
    dq[1:4n] .= q[4n+1:8n]
    # acceleration
    dq[4n+1:8n] .= inv(M)*(fg+fw(q)+fm(q,u(t))+fc(q,u(t)))
    # fix first 2 masses
    # dq[1:4] .= 0
    # dq[4n+1:4n+4] .= 0
    # return dq
end

uu(t) = zeros(3n-2)*t
octopus(similar(q0), q0, uu, 0)
##
tspan = (0,0.1)
ts = LinRange(tspan...,200)
prob = ODEProblem(octopus, q0, tspan, uu)
sol = solve(prob, Tsit5())
# sol = solve(prob, Rosenbrock23(autodiff=false); abstol=1e-8, reltol=1e-8)

##
using UnicodePlots
ts = LinRange(tspan...,200)
ts = LinRange(0, 0.005, 200)
qs = reduce(hcat, Lt(sol(t))[8:end] for t in ts);
show(lineplot(ts, qs'; width=160))
qs = reduce(hcat, Ft(sol(t), uu(t))[8:end] for t in ts);
show(lineplot(ts, qs'; width=160))
qs = reduce(hcat, P(sol(t), uu(t))[8:end] for t in ts);
show(lineplot(ts, qs'; width=160))
qs = reduce(hcat, fm(sol(t), uu(t)) for t in ts);
show(lineplot(ts, qs'; width=160))
##
qs = reduce(hcat, Lv(sol(t)) for t in ts);
show(lineplot(ts, qs'; width=160))
qs = reduce(hcat, FFd(sol(t), uu(t)) for t in ts);
show(lineplot(ts, qs'; width=160))
# names = reduce(vcat, ["xv$i"; "yv$i"; "xd$i"; "yd$i"] for i in 1:n)