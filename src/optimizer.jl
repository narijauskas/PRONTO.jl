# Types:
# Functor{NX,NU}() (simple version of closure + buffer)
# Trajectory = (x,u) = an efficient pair of interpolants, update together
# 



# B = functor((B,t)->fu!(B, x(t), u(t)), NX, NU)

function optimizer(NX,NU,T,)
    A = buffer(NX,NX)
    B = buffer(NX,NU)
    Q = buffer(NX,NX)
    R = buffer(NU,NU)
    S = buffer(NX,NU)
    Ko = buffer(NU,NX)

    P = buffer(NX,NX)
    pxx!(P, α(T)) # P(T) around unregulated trajectory
    P! = solve(ODEProblem(optimizer!, P, (T,0.0), (Ko,R,Q,A)))

    function _Ko()
        fu!(B, x(t), u(t))
        luu!(R, x(t), u(t))
        lxu!(S, x(t), u(t))

        copy!(Ko, R(t)\(S(t)'+B(t)'*P(t)))
        return Ko
    end
    return _Ko
end

function optimizer!(dP, P, (Ko,R,Q,A), t)
    dP .= -A(t)'*P - P*A(t) + Ko(P,t)'*R(t)*Ko(P,t) - Q(t)
end





# --------------- solve optimizer Ko --------------- #
tx = @elapsed begin
    PT = MArray{Tuple{NX,NX},Float64}(undef) # pxx!
    pxx!(PT, α(T)) # around unregulated trajectory

    
    # Ko = inv(R)\(S'+B'*P)
    Ko = Functor(NU,NX) do buf,t
        
    end
end
tinfo(i, "optimizer solved", tx)

fx!(A, x(t), u(t))
fu!(B, x(t), u(t))
lxx!(Q, x(t), u(t))
luu!(R, x(t), u(t))
lxu!(S, x(t), u(t))

