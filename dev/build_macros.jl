
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

@dynamics Model ex
build_f(Model, (x,u,t,θ)->ex)

macro dynamics(T, ex)
    :(build_f($T, (x,u,t,θ)->$ex))
end


# what goes n each macro is the body of a function(x,u,t,θ)