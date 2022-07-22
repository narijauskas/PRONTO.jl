

# fx = jacobian(x, f, x, u)
function jacobian(dx, f, args...; inplace = false)
    f_sym = Base.invokelatest(f, args...)
    fx_sym = cat(map(1:length(dx)) do i
        map(f_sym) do f
            derivative(f, dx[i])
        end
    end...; dims = ndims(f_sym)+1)

    # return eval(build_function(fx_sym, args...)[inplace ? 2 : 1])
    fx_ex = build_function(fx_sym, args...)[inplace ? 2 : 1]
    return @eval $fx_ex
end

# fxx = hessian(x, u, f, x, u)
hessian(dx1, dx2, f, args...; inplace=false) = jacobian(dx2, jacobian(dx1, f, args...), args...; inplace)
# function hessian(dx1, dx2, f, args...; inplace=false)
#     fx1 = jacobian(dx1, f, args...)
#     jacobian(dx2, Base.invokelatest(jacobian(dx1, f, args...)), args...; inplace)


#MAYBE:
# @kwdef
# struct Model{NX,NU}
# end

# structs dispatch on name, supertypes
# NamedTuples dispatch on the value of each field




# function autodiff(f=(x,u)->(x),l=(x,u)->(x),p=(x)->(x); NX=1, NU=1)
# function autodiff(f,l,p; NX,NU)
#     # model = MStruct()
#     # model.NX = NX
#     # model.NU = NU
#     model = (NX=NX, NU=NU)
#     autodiff!(model,f,l,p)
#     return model
# end

function autodiff(model,f,l,p)
    @variables x[1:model.NX] u[1:model.NU]

    return merge(model, (
        f = f, # NX
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

# alternatively? @unpack model NX NU ts fx! fu!

macro unpack(model)
    return esc(quote
        NX = $(model).NX
        NU = $(model).NU
        ts = $(model).ts
        f = $(model).f
        fx! = $(model).fx!
        fu! = $(model).fu!
        fxx! = $(model).fxx!
        fxu! = $(model).fxu!
        fuu! = $(model).fuu!
        l = $(model).l
        lx! = $(model).lx!
        lu! = $(model).lu!
        lxx! = $(model).lxx!
        lxu! = $(model).lxu!
        luu! = $(model).luu!
        p = $(model).p
        px! = $(model).px!
        pxx! = $(model).pxx!
        x0 = $(model).x0
        Qr = $(model).Qr
        Rr = $(model).Rr
        iRr = $(model).iRr
    end)
end



# model.f = f
    # model.fx! = jacobian(x,f,x,u; inplace=true)
    # model.fu! = jacobian(u,f,x,u; inplace=true)
    # model.fxx! = hessian(x,x,f,x,u; inplace=true)
    # model.fxu! = hessian(x,u,f,x,u; inplace=true)
    # model.fuu! = hessian(u,u,f,x,u; inplace=true)
    # model.l = l
    # model.lx! = jacobian(x,l,x,u; inplace=true)
    # model.lu! = jacobian(u,l,x,u; inplace=true)
    # model.lxx! = hessian(x,x,l,x,u; inplace=true)
    # model.lxu! = hessian(x,u,l,x,u; inplace=true)
    # model.luu! = hessian(u,u,l,x,u; inplace=true)
    # model.p = p
    # model.px! = jacobian(x,p,x; inplace=true)
    # model.pxx! = hessian(x,x,p,x; inplace=true)

    # return model