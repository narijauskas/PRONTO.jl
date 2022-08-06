function inplace(f, args...; inplace=true)
    f_sym = cat(Base.invokelatest(f, args...); dims=1)
    f_ex = build_function(f_sym, args...)[inplace ? 2 : 1]
    return @eval $f_ex
end

# function build_fxn(f_sym, args...; inplace = false)
#     f_ex = build_function(f_sym, args...)[inplace ? 2 : 1]
#     return @eval $f_ex
# end

# function build_sparse(f_sym, args...; inplace = false)
#     f_ex = build_function(sparse(f_sym), args...)[inplace ? 2 : 1]
#     return @eval $f_ex
# end

# fx = jacobian(x, f, x, u)
function jacobian(dx, f, args...; inplace = false)
    f_sym = Base.invokelatest(f, args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(f_sym) do f
            derivative(f, dx[i])
        end
    end...; dims = ndims(f_sym)+1)
    fx_ex = build_function(fx_sym, args...)[inplace ? 2 : 1]
    return @eval $fx_ex
end


# fxx = hessian(x, u, f, x, u)
hessian(dx1, dx2, f, args...; inplace = false) = jacobian(dx2, jacobian(dx1, f, args...), args...; inplace)


function _autodiff(model,f,l,p)
    @variables x[1:model.NX] u[1:model.NU]

    return merge(model, (
        f = f, # NX
        fx = jacobian(x,f,x,u; inplace=false), # NX,NX #NOTE: for testing
        fx! = jacobian(x,f,x,u; inplace=true), # NX,NX
        fu! = jacobian(u,f,x,u; inplace=true), # NX,NU
        fxx! = hessian(x,x,f,x,u; inplace=true), # NX,NX,NX
        fxu! = hessian(x,u,f,x,u; inplace=true), # NX,NX,NU
        fuu! = hessian(u,u,f,x,u; inplace=true), # NX,NU,NU
        l = l, # 1
        lx! = jacobian(x,l,x,u; inplace=true), # NX
        lu! = jacobian(u,l,x,u; inplace=true), # NU
        lxx! = hessian(x,x,l,x,u; inplace=true), # NX,NX
        lxu! = hessian(x,u,l,x,u; inplace=true), # NX,NU
        luu! = hessian(u,u,l,x,u; inplace=true), # NU,NU
        p = p, # 1
        px! = jacobian(x,p,x; inplace=true), # NX
        pxx! = hessian(x,x,p,x; inplace=true), # NX,NX
    ))

    return model
end



function autodiff(model,f,l,p)
    @variables x[1:model.NX] u[1:model.NU] λ[1:model.NX]

    # Lagrangian
    L = (x,u,λ)->(collect(l(x,u)) .+ collect(λ)'collect(f(x,u)))

    return merge(model, (
        f = f, # NX
        f! = inplace(f,x,u), # NX
        fx = jacobian(x,f,x,u; inplace=false), # NX,NX #NOTE: for testing
        fx! = jacobian(x,f,x,u; inplace=true), # NX,NX
        fu! = jacobian(u,f,x,u; inplace=true), # NX,NU
        fxx! = hessian(x,x,f,x,u; inplace=true), # NX,NX,NX
        fxu! = hessian(x,u,f,x,u; inplace=true), # NX,NX,NU
        fuu! = hessian(u,u,f,x,u; inplace=true), # NX,NU,NU
        l = l, #inplace(l,x,u), # 1
        lx! = jacobian(x,l,x,u; inplace=true), # NX
        lu! = jacobian(u,l,x,u; inplace=true), # NU
        lxx! = hessian(x,x,l,x,u; inplace=true), # NX,NX
        lxu! = hessian(x,u,l,x,u; inplace=true), # NX,NU
        luu! = hessian(u,u,l,x,u; inplace=true), # NU,NU
        p = p, #inplace(p,x), # 1
        px! = jacobian(x,p,x; inplace=true), # NX
        pxx! = hessian(x,x,p,x; inplace=true), # NX,NX
        L = L, #inplace(L,x,u,λ), # 1
        Lx! = jacobian(x,L,x,u,λ; inplace=true), # NX
        Lu! = jacobian(u,L,x,u,λ; inplace=true), # NU
        Lxx! = hessian(x,x,L,x,u,λ; inplace=true), # NX,NX
        Lxu! = hessian(x,u,L,x,u,λ; inplace=true), # NX,NU
        Luu! = hessian(u,u,L,x,u,λ; inplace=true), # NU,NU
    ))

    return model
end
