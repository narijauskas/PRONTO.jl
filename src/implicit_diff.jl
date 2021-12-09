Jx(f, X, U) = t->jacobian(x->f(x, U(t)), X(t))
Ju(f, X, U) = t->jacobian(u->f(X(t), u), U(t))


function Hxx(f, X, U)
    nx = length(X(0))
    hess = t->jacobian(X(t)) do xx
        jacobian(x->f(x, U(t)), xx)
    end
    return t->permutedims(reshape(hess(t), nx, nx, nx), (2,3,1))
end


function Huu(f, X, U)
    nu = length(U(0))
    nx = length(X(0))
    hess = t->jacobian(U(t)) do uu
        jacobian(u->f(X(t), u), uu)
    end
    return t->permutedims(reshape(hess(t), nx, nu, nu), (2,3,1))
end


function Hxu(f, X, U)
    nu = length(U(0))
    nx = length(X(0))
    hess = t->jacobian(U(t)) do u
        jacobian(x->f(x, u), X(t))
    end
    return t->permutedims(reshape(hess(t), nx, nu, nx), (2,3,1))
end