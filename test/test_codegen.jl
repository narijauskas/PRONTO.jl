using Test
using PRONTO
using LinearAlgebra
using StaticArrays
using Base: @kwdef


## ----------------------------------- define the model ----------------------------------- ##

@kwdef struct TwoSpin <: Model{4,1}
    kr::Float64 = 1.0
    kq::Float64 = 1.0
end

@define_f TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

# @define_l TwoSpin begin
#     Rl = [0.01;;]
#     1/2*u'*Rl*u
# end

# @define_m TwoSpin begin
#     Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
#     1/2*x'*Pl*x
# end

# @define_Q TwoSpin kq*I(4)
# @define_R TwoSpin kr*I(1)
# resolve_model(TwoSpin)

## ----------------------------------- test model derivation ----------------------------------- ##

θ = TwoSpin() # instantiate a new model
t = 0.0
u = @SVector [0.1]
x = @SVector [0.5, 0.6, 0.7, 0.8]


@testset "dynamics" begin
    @test PRONTO.f(θ,x,u,t) == [
        x[3] - u[1]*x[2];
        -x[4] + u[1]*x[1];
        -x[1] - u[1]*x[4];
        x[2] + u[1]*x[3];
    ]
end

@testset "dynamics jacobians" begin
    @test PRONTO.fx(θ,x,u,t) == [
        0 -u[1] 1 0;
        u[1] 0 0 -1;
        -1 0 0 -u[1];
        0 1 u[1] 0;
    ]

    @test PRONTO.fu(θ,x,u,t) == [
        -x[2];
        x[1];
        -x[4];
        x[3];;
    ]
end

@testset "dynamics hessians" begin
    @test PRONTO.fxx(θ,x,u,t) == zeros(4,4,4)

    @test PRONTO.fxu(θ,x,u,t) == Float64[
        0 -1 0 0;
        1 0 0 0;
        0 0 0 -1;
        0 0 1 0;;;
    ]

    @test PRONTO.fuu(θ,x,u,t) == zeros(4,1,1)
end

