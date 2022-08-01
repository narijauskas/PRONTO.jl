# --------------------------------- optimizer Ko --------------------------------- #

function optimizer(x,u,PT,model)
    fx! = model.fx!; fu! = model.fu!;
    lxx! = model.lxx!; luu! = model.luu!; lxu! = model.lxu!;
    NX = model.NX; NU = model.NU; T = model.T;

    A = functor(@closure((A,t) -> fx!(A,x(t),u(t))), buffer(NX,NX))
    B = functor(@closure((B,t) -> fu!(B,x(t),u(t))), buffer(NX,NU))
   
    Q = functor(@closure((Q,t) -> lxx!(Q,x(t),u(t))), buffer(NX,NX))
    R = functor(@closure((R,t) -> luu!(R,x(t),u(t))), buffer(NU,NU))
    S = functor(@closure((S,t) -> lxu!(S,x(t),u(t))), buffer(NX,NU))

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


