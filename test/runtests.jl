using PRONTO, Test, SafeTestsets

begin
    @safetestset "say hello" begin include("test_file.jl") end
end