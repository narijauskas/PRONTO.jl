using PRONTO
using Test
using StaticArrays
using LinearAlgebra
using Base: @kwdef

NX = 4
NU = 1
NΘ = 2

@kwdef struct TwoSpin{T} <: PRONTO.Model{NX,NU,NΘ}
    kr::T = 1.0
    kq::T = 1.0
end

## ----------------------------------- generate solver kernel ----------------------------------- ##

@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

@stage_cost TwoSpin begin
    Rl = [0.01;;]
    1/2 * u'*Rl*u
end

@terminal_cost TwoSpin begin
    Pl = [0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 1]
    1/2*x'*Pl*x
end

@regulatorQ TwoSpin θ.kq*I(NX)
@regulatorR TwoSpin θ.kr*I(NU)
@lagrangian TwoSpin


# overwrite default behavior of Pf
PRONTO.Pf(α,μ,tf,θ::TwoSpin{T}) where T = SMatrix{4,4,T}(I(4))


## ----------------------------------- test ----------------------------------- ##

# ex = PRONTO.f(x,u,t,θ)
# PRONTO.fx(x,u,t,θ)
# PRONTO.fu(x,u,t,θ)


# symbolic(:x, 1:NX)

θ = TwoSpin{Float64}()
t = 0.0
u = @SVector [0.1]
x = @SVector [0.5, 0.6, 0.7, 0.8]

##

@testset "dynamics" begin
    @test PRONTO.f(x,u,t,θ) == [
        x[3] - u[1]*x[2];
        -x[4] + u[1]*x[1];
        -x[1] - u[1]*x[4];
        x[2] + u[1]x[3];
    ]
end

@testset "dynamics derivatives" begin
    @test PRONTO.fx(x,u,t,θ) == [
        0 -u[1] 1 0;
        u[1] 0 0 -1;
        -1 0 0 -u[1];
        0 1 u[1] 0;
    ]

    @test PRONTO.fu(x,u,t,θ) == [
        -x[2];
        x[1];
        -x[4];
        x[3];;
    ]

    @test PRONTO.fxx(x,u,t,θ) == zeros(4,4,4)

    @test PRONTO.fxu(x,u,t,θ) == Float64[
        0 -1 0 0;
        1 0 0 0;
        0 0 0 -1;
        0 0 1 0;;;
    ]

    @test PRONTO.fuu(x,u,t,θ) == zeros(4,1,1)
end
