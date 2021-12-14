using PRONTO
using LinearAlgebra

## ------------------------------ USER INPUTS ------------------------------ ## 
 
# define dynamics
function fxn(x, u)
    # parameters:
    l = 1; g = 9.8;
    # dynamics:
    return [
        x[2],
        (g/l)*sin(x[1]) - (u/l) * cos(x[1])]
end

T = 10 # final time

# desired trajectory
xd(t) = t->[0.0; 0.0]
ud(t) = t->[0.0]
ξd = Trajectory(xd, ud)

# equilibrium trajectory
xe(t) = t->[0.0; 0.0]
ue(t) = t->[0.0]
ξeqb = Trajectory(xe, ue)

# regulator parameters
Qr = I(2)
Rr = I(1)

# cost parameters
Qc = I(2)
Rc = I(1)

# cost functional
l, m = build_LQ_cost(ξd, Qc, Rc, P1, T)


## ------------------------------ DO PRONTO STUFF ------------------------------ ## 

