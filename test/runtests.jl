using PRONTO, Test, SafeTestsets

begin
    @safetestset "say hello" begin include("test_file.jl") end
    @safetestset "timeseries" begin include("test_timeseries.jl") end
end