using PRONTO, Test

T = 0:0.01:10

# test construction
U1 = Timeseries(t->sin(t), T)
U2 = Timeseries(t->[sin(t)], T)
U3 = Timeseries(t->[sin(t); cos(t)], T)
U4 = Timeseries(t->[sin(t);;cos(t)], T)
U5 = Timeseries(t->[sin(t); cos(t);;], T)
U6 = Timeseries(t->[cos(t); sin(t);;], T)


# test types
@test U1(0) isa Float64
@test U2(0) isa Vector{Float64}
@test U3(0) isa Vector{Float64}
@test U4(0) isa Matrix{Float64}
@test U5(0) isa Matrix{Float64}
@test U6(0) isa Matrix{Float64}


@test Timeseries{typeof(U1(0))} == typeof(U1)
@test Timeseries{typeof(U2(0))} == typeof(U2)
@test Timeseries{typeof(U3(0))} == typeof(U3)
@test Timeseries{typeof(U4(0))} == typeof(U4)
@test Timeseries{typeof(U5(0))} == typeof(U5)
@test Timeseries{typeof(U6(0))} == typeof(U6)



# test size
@test () == size(U1(0))
@test (1,) == size(U2(0))
@test (2,) == size(U3(0))
@test (1,2) == size(U4(0))
@test (2,1) == size(U5(0))
@test (2,1) == size(U6(0))





# addition
UA = Timeseries(t->U5(t)+U6(t), T)
@test UA(0) == U5(0)+U6(0)

# scalar multiplication
UM = Timeseries(t->2*U5(t), T)
@test UM(0) == 2*U5(0)

# arbitrary function
UX = Timeseries(t->sin.(U6(t)), T)
@test UX(0) == sin.(U6(0))
