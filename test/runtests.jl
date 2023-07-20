using PRONTO, Test, SafeTestsets

begin
    @safetestset "code generation: TwoSpin" begin include("test_codegen.jl") end
end