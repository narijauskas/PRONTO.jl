# --------------------------------- optimizer Ko --------------------------------- #

function optimizer(A,B,Q,R,S,PT,NX,NU,T)
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


