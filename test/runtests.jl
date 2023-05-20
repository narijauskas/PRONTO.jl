using PRONTO, Test, SafeTestsets

begin
    @safetestset "code generation: TwoSpin" begin include("codegen_twospin.jl") end
end