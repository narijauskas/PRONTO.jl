import Base: +, -, *
export Trajectory 

mutable struct Trajectory
    x :: Function
    u :: Function
    Trajectory(x::Function, u::Function) = new(x,u)
end

Trajectory(x::Function, u) = Trajectory(x, t->u(t))
Trajectory(x, u::Function) = Trajectory(t->x(t), u)
Trajectory(x, u) = Trajectory(t->x(t), t->u(t))
# Trajectory(sol1, sol2) = Trajectory(
#     (t) -> sol1(t),
#     (t) -> sol2(t)
# )

# Operators on two trajectories
(+)(ξ1::Trajectory, ξ2::Trajectory) = Trajectory(
    (t) -> (ξ1.x(t) + ξ2.x(t)),
    (t) -> (ξ1.u(t) + ξ2.u(t)))

(-)(ξ1::Trajectory, ξ2::Trajectory) = Trajectory(
    (t) -> (ξ1.x(t) - ξ2.x(t)),
    (t) -> (ξ1.u(t) - ξ2.u(t)))

(*)(ξ1::Trajectory, ξ2::Trajectory) = Trajectory( # maybe call this .* instead?
    (t) -> (ξ1.x(t) * ξ2.x(t)),
    (t) -> (ξ1.u(t) * ξ2.u(t)))

# Operators on a trajectory and tuple of scalars
(*)(γ::Tuple{Real, Real}, ξ::Trajectory) = Trajectory(
    (t) -> (ξ.x(t) * γ[1]),
    (t) -> (ξ.u(t) * γ[2]))

(*)(ξ::Trajectory, γ::Tuple{Real, Real}) = γ*ξ

#MAYBE: support single scalar multiplication
