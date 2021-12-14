Jx(f, X, U) = t->jacobian(x->f(x, U(t)), X(t))
Ju(f, X, U) = t->jacobian(u->f(X(t), u), U(t))

# Jx(f, X, U) = jacobian(x->f(x, U), X)
# t->Jx(f, X(t), U(t))

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

Jx(f, ξ) = Jx(f, ξ.x, ξ.u)
Ju(f, ξ) = Ju(f, ξ.x, ξ.u)

Hxx(f, ξ) = Hxx(f, ξ.x, ξ.u)
Huu(f, ξ) = Huu(f, ξ.x, ξ.u)
Hxu(f, ξ) = Hxu(f, ξ.x, ξ.u)
