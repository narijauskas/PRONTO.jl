using PRONTO, Test

T = 0:0.01:10

U1 = Timeseries(t->sin(t), T)
U2 = Timeseries(t->[sin(t)], T)
U3 = Timeseries(t->[sin(t); cos(t)], T)
U4 = Timeseries(t->[sin(t);;cos(t)], T)
U5 = Timeseries(t->[sin(t); cos(t);;], T)

@test U1(0) isa Float64
@test U2(0) isa Vector{Float64}
@test U3(0) isa Vector{Float64}
@test U4(0) isa Matrix{Float64}
@test U5(0) isa Matrix{Float64}

@test () == size(U1(0))
@test (1,) == size(U2(0))
@test (2,) == size(U3(0))
@test (1,2) == size(U4(0))
@test (2,1) == size(U5(0))

