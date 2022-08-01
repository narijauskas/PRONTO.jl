# --------------------------------- optimizer Ko --------------------------------- #

function optimizer(x,u,PT,model)
    NX = model.NX; NU = model.NU; T = model.T;
    fx! = model.fx!; _A = buffer(NX,NX)
    A = @closure (t)->(fx!(_A,x(t),u(t)); return _A)
    fu! = model.fu!; _B = buffer(NX,NU)
    B = @closure (t)->(fu!(_B,x(t),u(t)); return _B)
    lxx! = model.lxx!; _Q = buffer(NX,NX)
    Q = @closure (t)->(lxx!(_Q,x(t),u(t)); return _Q)
    luu! = model.luu!; _R = buffer(NU,NU)
    R = @closure (t)->(luu!(_R,x(t),u(t)); return _R)
    lxu! = model.lxu!; _S = buffer(NX,NU)
    S = @closure (t)->(lxu!(_S,x(t),u(t)); return _S)

    P! = solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S)))
    P = functor((P,t)->P!(P,t), buffer(NX,NX))
    # Ko = R\(S'+B'*P) # maybe inv!()
    Ko = buffer(NU,NX)
    function _Ko(t)
        copy!(Ko, R(t)\(S(t)'+B(t)'*P(t)))
        return Ko
    end
    return _Ko
end

function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end


