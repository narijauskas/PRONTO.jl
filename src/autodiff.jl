using Symbolics
using Symbolics: derivative
#TODO: define Dx() operators
#MAYBE: define Jx(), Hx() operators


function jacobian(dx, f, args...; inplace = false)
    f_sym = f(args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(1:length(f_sym)) do j
            derivative(f_sym[j], dx[i])
        end
    end...; dims = ndims(f_sym)+1)

    return eval(build_function(fx_sym, args...)[inplace ? 2 : 1])
end
