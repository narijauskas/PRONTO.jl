using PRONTO, Test, SafeTestsets

begin
    @safetestset "timeseries" begin include("test_timeseries.jl") end
    @safetestset "autodiff" begin include("test_autodiff.jl") end
end