using PRONTO
using PRONTO: clean
using MacroTools: prettify
using Symbolics
using Test

# @testset "symbolics" begin
#     #TODO: function tracing
#     #TODO: jacobians
# end

T = TwoSpin

@testset "kernel generation" begin
    # test generation of function body from a symbolic kernel: Num[] -> Expr[]

    x = first(@variables x[1:nx(T)])
    u = first(@variables u[1:nu(T)])

    sym = [
        x[3] - u[1]*x[2]
        u[1]*x[1] - x[4]
        -x[1] - u[1]*x[4]
        u[1]*x[3] + x[2]
    ]

    test_kernel = [
        :(out[1] = (-1 * u[1]) * x[2] + x[3])
        :(out[2] = u[1] * x[1] + -1 * x[4])
        :(out[3] = -1 * x[1] + (-1 * u[1]) * x[4])
        :(out[4] = u[1] * x[3] + x[2])
    ]

    @test (PRONTO.def_kernel(sym).|>prettify) == test_kernel
end

@testset "inplace method generation" begin

    body = [
        :(out[1] = (-1 * u[1]) * x[2] + x[3])
        :(out[2] = u[1] * x[1] + -1 * x[4])
        :(out[3] = -1 * x[1] + (-1 * u[1]) * x[4])
        :(out[4] = u[1] * x[3] + x[2])
    ]

    test_inplace = quote
        function PRONTO.f!(out, x, u, t, θ::TwoSpin)
            @inbounds begin
                out[1] = (-1 * u[1]) * x[2] + x[3]
                out[2] = u[1] * x[1] + -1 * x[4]
                out[3] = -1 * x[1] + (-1 * u[1]) * x[4]
                out[4] = u[1] * x[3] + x[2]
            end
            return out
        end
    end |> clean |> string

    gen_inplace = PRONTO.def_inplace(:f, T, body, :x, :u, :t) |> string
    @test isequal(gen_inplace, test_inplace)
end


@testset "generic method generation" begin

    test_generic = quote
        function PRONTO.f(x, u, t, θ::TwoSpin)
            out = MVector{4, Float64}(undef)
            PRONTO.f!(out, x, u, t, θ)
            return SVector{4, Float64}(out)
        end
    end |> clean |> string

    gen_generic = PRONTO.def_generic(:f, T, Size(nx(T)), :x, :u, :t) |> string
    @test isequal(gen_generic, test_generic)
end

@testset "symbolic method generation" begin

    test_symbolic = quote
        function PRONTO.f(x, u, t, θ::SymModel{TwoSpin})
            out = Array{Num}(undef, 4)
            PRONTO.f!(out, x, u, t, θ)
            return SArray{Tuple{4}, Num}(out)
        end
    end |> clean |> string

    gen_symbolic = PRONTO.def_symbolic(:f, T, Size(nx(T)), :x, :u, :t) |> string
    @test isequal(gen_symbolic, test_symbolic)
end

#TODO: trace -> build -> trace -> build