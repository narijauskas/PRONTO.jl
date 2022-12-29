#=
implementation of PRONTO for the model presented in Yekutieli et al 2005
Dynamic Model of the Octopus Arm. I. Biomechanics of the Octopus Reaching Movement
=#

# parameters
# Arm width is 15 mm at thebase and 1 mm at the tip.
# Arm length is 240 mm and the depthat every point equals the width at that point and remainsconstant throughout the simulation
ρ_arm = 1042 # kg/m^3, arm density
ρ_sea = 1022 # kg/m^3, seawater density

# states:
# q = ...
# size 4n

# inputs: u = a(t) ∈ [0,1]
# size 3n-2 or n-1 depending on assumptions

# model dynamics
q̈ = inv(M)*(fm(x,u)+fg(x)+fw(x)+fc(x))

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

#Fv is piecewise... use atan smooth approx or see Zajac 1989
# Fv(v⋆) = 
# 0 if v⋆ ≤ -1
# 1+v⋆ if -1 < v⋆ < 0.8
# 1.8 if 0.8 ≥ v⋆
# where v⋆ = v/vmax
# vmax = 10*lm / sec (10 muscle lengths/sec)

# Fa and Fp are data driven normalized curves... 🤮
# scaled by maximal stress: 8e4 N/m^2



# fg: gravity/buoyancy
fg(x) = (ρ_arm - ρ_sea)*V_arm(x)*g
# V_arm is a 4Nx4N matrix, and g must be vectorized somehow

# fw: drag forces
c_per = 1.013 # dimensionless perpendicular drag (experimental)
c_tan = 0.0256 # dimensionless tangential drag (experimental)


# fc: hydrostat constraint
fc(x) = C*p(x)
# C = ?? #TODO: matrix from lagrange
p(x) = inv(G*inv(M)*C)*(γ-G*inv(M)*(fm(x) + fg(x) + fw(x)))
#TODO: understand G matrix, which is a function of size
#TODO: size vector



# cost functions for reaching
# stage cost: ??
# terminal cost: extended arm?

# regultor settings
# - equal priority
# - prioritize medial muscles
# - prioritize lateral muscles

# comparison of optimal trajectory to open-loop propogating wave input
# - generate neural activation pattern u_neural(t)