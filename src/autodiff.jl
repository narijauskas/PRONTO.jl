

# fx = jacobian(x, f, x, u)
function jacobian(dx, f, args...; inplace = false)
    f_sym = f(args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(f_sym) do f
            derivative(f, dx[i])
        end
    end...; dims = ndims(f_sym)+1)

    return eval(build_function(fx_sym, args...)[inplace ? 2 : 1])
end


# fxx = hessian(x, u, f, x, u)
hessian(dx1, dx2, f, args...; inplace=false) = jacobian(dx2, jacobian(dx1, f, args...), args...; inplace)