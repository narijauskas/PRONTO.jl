# module MStructs
export MStruct

# An object which mimics the convenient behavior of MATLAB structs.
# ie, a Dict with dot access. Not thread-safe. Not fast. To use:
# foo = MStruct()
# foo.bar = 1
struct MStruct
    data::Dict{Symbol,Any}
    MStruct() = new(Dict{Symbol,Any}())
end

Base.setproperty!(x::MStruct, name::Symbol, v) = setindex!(getfield(x, :data), v, name)
Base.getproperty(x::MStruct, name::Symbol) = getindex(getfield(x, :data), name)
Base.length(x::MStruct) = length(getfield(x, :data))
Base.show(io::IO, x::MStruct) = show(io, getfield(x, :data))

# end #module