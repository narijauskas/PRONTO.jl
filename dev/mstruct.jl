# an object which (almost) mimics the convenient behavior of MATLAB structs
# ie, a Dict with dot access
# not thread-safe

struct MStruct
    data::Dict{Symbol,Any}
end

MStruct() = MStruct(Dict{Symbol,Any}())

# getfield(x, :data) is needed because we're overwriting the method responsible for the behavior of x.data
Base.setproperty!(x::MStruct, name::Symbol, v) = setindex!(getfield(x, :data), v, name)
Base.getproperty(x::MStruct, name::Symbol) = getindex(getfield(x, :data), name)
Base.length(x::MStruct) = length(getfield(x, :data))
Base.show(io::IO, x::MStruct) = show(io, getfield(x, :data))
