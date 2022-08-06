# --------------------------------- optimizer Ko --------------------------------- #

function optimizer(x,u,PT,model)
    NX = model.NX; NU = model.NU; T = model.T;
    fx! = model.fx!; _A = Buffer{Tuple{NX,NX}}()
    A = @closure (t)->(fx!(_A,x(t),u(t)); return _A)
    fu! = model.fu!; _B = Buffer{Tuple{NX,NU}}()
    B = @closure (t)->(fu!(_B,x(t),u(t)); return _B)
    lxx! = model.lxx!; _Q = Buffer{Tuple{NX,NX}}()
    Q = @closure (t)->(lxx!(_Q,x(t),u(t)); return _Q)
    luu! = model.luu!; _R = Buffer{Tuple{NU,NU}}()
    R = @closure (t)->(luu!(_R,x(t),u(t)); return _R)
    lxu! = model.lxu!; _S = Buffer{Tuple{NX,NU}}()
    S = @closure (t)->(lxu!(_S,x(t),u(t)); return _S)

    # try
    #     P! = solve(ODEProblem(optimizer!, collect(PT), (T,0.0), (A,B,Q,R,S)))
    # catch e
    #     P! = solve(ODEProblem(optimizer!, collect(PT), (T,0.0), (A,B,Q,R,S)))
    # end

    _P = Buffer{Tuple{NX,NX}}()
    P! = solve(ODEProblem(optimizer!, collect(PT), (T,0.0), (A,B,Q,R,S)))
    P = @closure (t)->(P!(_P,t); return _P)

    # P = functor((_P,t)->P!(_P,t), Buffer{Tuple{NX,NX}}())
    # Ko = R\(S'+B'*P) # maybe inv!()
    _Ko = Buffer{Tuple{NU,NX}}()
    function Ko(t)
        copy!(_Ko, R(t)\(S(t)'+B(t)'*P(t)))
        return _Ko
    end
    return Ko
end

struct InstabilityException <: Exception end

function optimizer!(dP, P, (A,B,Q,R,S), t)
    Ko = R(t)\(S(t)'+B(t)'*P)
    # Ko = R(t)\B(t)'*P
    #TODO: if maxeig(P)>1e8 throw(InstabilityException)
    dP .= -A(t)'*P - P*A(t) + Ko'*R(t)*Ko - Q(t)
end

# foo(x) = iseven(x) ? x : false
# bar(x) = iseven(x) ? x : x+1
# bar(x::Bool) = iseven(x) ? x : !x

# @code_warntype foo(1)
# @code_warntype foo(2)