
# this code:

function dynamics(x,u,t,θ)
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

PRONTO.build_f(TwoSpin, dynamics)

# is the same as this code:

PRONTO.build_f(TwoSpin, (x,u,t,θ) -> begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end)


# our goal is to make it look like this:
@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

# in other words, the macro
@dynamics T ex
# remaps to:
build_f(T, (x,u,t,θ)->ex)

macro dynamics(T, ex)
    :(build_f($T, (x,u,t,θ)->$ex))
end


# model dynamics are described by an anonymous function
(x,u,t,θ) -> begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end



# in conclusion, when we write:
@dynamics TwoSpin begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end

# it expands to:
PRONTO.define_dynamics(TwoSpin, (x,u,t,θ) -> begin
    H0 = [0 0 1 0;0 0 0 -1;-1 0 0 0;0 1 0 0]
    H1 = [0 -1 0 0;1 0 0 0;0 0 0 -1;0 0 1 0]
    (H0 + u[1]*H1)*x
end)

# which generates the following definitions:
function PRONTO.f!(out, x, u, t, θ::TwoSpin)
    @inbounds begin
            out[1] = (-1 * u[1]) * x[2] + x[3]
            out[2] = u[1] * x[1] + -1 * x[4]
            out[3] = -1 * x[1] + (-1 * u[1]) * x[4]
            out[4] = u[1] * x[3] + x[2]
        end
    return nothing
end

function PRONTO.f(x, u, t, θ::TwoSpin)
    out = (MVector{4, Float64})(undef)
    @inbounds begin
            out[1] = (-1 * u[1]) * x[2] + x[3]
            out[2] = u[1] * x[1] + -1 * x[4]
            out[3] = -1 * x[1] + (-1 * u[1]) * x[4]
            out[4] = u[1] * x[3] + x[2]
        end
    return (SVector{4, Float64})(out)
end

function PRONTO.fx(x, u, t, θ::TwoSpin)
    out = (MMatrix{4, 4, Float64})(undef)
    @inbounds begin
            out[1] = 0
            out[2] = u[1]
            out[3] = -1
            out[4] = 0
            out[5] = -1 * u[1]
            out[6] = 0
            out[7] = 0
            out[8] = 1
            out[9] = 1
            out[10] = 0
            out[11] = 0
            out[12] = u[1]
            out[13] = 0
            out[14] = -1
            out[15] = -1 * u[1]
            out[16] = 0
        end
    return (SMatrix{4, 4, Float64})(out)
end

function PRONTO.fu(x, u, t, θ::TwoSpin)
    out = (MMatrix{4, 1, Float64})(undef)
    @inbounds begin
            out[1] = -1 * x[2]
            out[2] = x[1]
            out[3] = -1 * x[4]
            out[4] = x[3]
        end
    return (SMatrix{4, 1, Float64})(out)
end














using Base: @kwdef

@kwdef struct Foo{T}
    x::T = 1
    y::T = 2.5
end