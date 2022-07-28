# --------------------------------- optimizer Ko --------------------------------- #

function optimizer(NX,NU,T,x,u,α,fx!,fu!,lxx!,luu!,lxu!,pxx!)
    A = functor((A,t) -> fx!(A,x(t),u(t)), buffer(NX,NX))
    B = functor((B,t) -> fu!(B,x(t),u(t)), buffer(NX,NU))
    Q = functor((Q,t) -> lxx!(Q,x(t),u(t)), buffer(NX,NX))
    R = functor((R,t) -> luu!(R,x(t),u(t)), buffer(NU,NU))
    S = functor((S,t) -> lxu!(S,x(t),u(t)), buffer(NX,NU))

    PT = buffer(NX,NX); pxx!(PT, α(T)) # P(T) around unregulated trajectory
    P! = solve(ODEProblem(optimizer!, PT, (T,0.0), (A,B,Q,R,S)))
    P = functor((P,t)->P!(P,t), buffer(NX))

    # Ko = R\(S'+B'*P) # maybe inv!()
    Ko = buffer(NU,NX)
    function _Ko()
        copy!(Ko, R(t)\(S(t)'+B(t)'*P(t)))
        return Ko
    end
    return _Ko
end

function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end


