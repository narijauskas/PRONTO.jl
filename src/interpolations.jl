using DataInterpolations

T = 10
t = 0:0.1:T

U = LinearInterpolation(map(τ->[τ;;;], t),t)

u0 = U(0.0)


