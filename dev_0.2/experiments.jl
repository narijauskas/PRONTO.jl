macro track(ex, name)
    return quote
        tx = @elapsed begin
            $esc($ex)
        end
        push!(stats[$name], tx)
        tinfo(i, $name, tx)
    end
end

# do things like:
@track begin
    Kr = regulator(α,μ,model)
end :regulator

# instead of:
tx = @elapsed begin
    Kr = regulator(α,μ,model)
end
push!(stats[:regulator], tx)
tinfo(i, "regulator solved", tx)

# --------------------------------- simultaneous optimizer/costate --------------------------------- #

function optimizer_costate(x,u,PT,rT,model)
    NX = model.NX; NU = model.NU; T = model.T;
    fx! = model.fx!; _A = buffer(NX,NX)
    A = @closure (t)->(fx!(_A,x(t),u(t)); return _A)
    fu! = model.fu!; _B = buffer(NX,NU)
    B = @closure (t)->(fu!(_B,x(t),u(t)); return _B)
    lx! = model.lx!; _a = buffer(NX)
    a = @closure (t)->(lx!(_a,x(t),u(t)); return _a)
    lu! = model.lu!; _b = buffer(NU)
    b = @closure (t)->(lu!(_b,x(t),u(t)); return _b)
    lxx! = model.lxx!; _Q = buffer(NX,NX)
    Q = @closure (t)->(lxx!(_Q,x(t),u(t)); return _Q)
    luu! = model.luu!; _R = buffer(NU,NU)
    R = @closure (t)->(luu!(_R,x(t),u(t)); return _R)
    lxu! = model.lxu!; _S = buffer(NX,NU)
    S = @closure (t)->(lxu!(_S,x(t),u(t)); return _S)

    P! = solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S)))
    P = functor((P,t)->P!(P,t), buffer(NX,NX))

    _Ko = buffer(NU,NX)
    Ko = @closure (t)->(copy!(_Ko, R(t)\(S(t)'+B(t)'*P(t))); return _Ko)
    
    r! = solve(ODEProblem(costate_dynamics!, rT, (T,0.0), (A,B,a,b,Ko)))
    r = functor((r,t)->r!(r,t), buffer(NX))

    _vo = buffer(NU)
    _iR = buffer(NU,NU)

    function Kovo(t)
        # copy!(_iR, R(t))
        # inv!(cholesky!(_iR))
        _iR .= inv(R(t))
        mul!(_Ko, _iR, (S(t)'+B(t)'*P(t)))
        _iR .*= -1
        mul!(_vo, _iR, (B(t)'*r(t)+b(t)))
        return (_Ko, _vo)
    end

    return Kovo
end


#=
tx = @elapsed begin
    Kovo = optimizer_costate(x,u,PT(α),rT(α),model)
end
push!(stats[:optimizer], tx/2)
push!(stats[:costate], tx/2)
tinfo(i, "optimizer and costate found", tx)

tx = @elapsed begin
    _z = search_z(x,u,Kovo,model)
    update!(z, _z)
    _v = search_v(z,Kovo,model)
    update!(v, _v)
end
push!(stats[:search_dir], tx)
tinfo(i, "search direction found", tx)

=#