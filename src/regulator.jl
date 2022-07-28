
# --------------------------------- regulator --------------------------------- #

function regulator(NX,NU,T,α,μ,fx!,fu!,iRr,Rr,Qr)
    # Ar = functor((Ar,t) -> fx!(Ar,α(t),μ(t)), buffer(NX,NX))
    # Br = functor((Br,t) -> fu!(Br,α(t),μ(t)), buffer(NX,NU))

    Ar = buffer(NX,NX) 
    Br = buffer(NX,NU)
    Kr = buffer(NU,NX)
    PT = collect(I(NX))
    Pr! = solve(ODEProblem(riccati!, PT, (T,0.0), (α,μ,fx!,fu!,Ar,Br,Rr,Qr,Kr)))
    Pr = buffer(NX,NX) # this setup allows in-place update: Pr!(Pr, t)
    iRrBr = buffer(NU,NX)
    function _Kr(t)
        fu!(Br, α(t), μ(t))
        Pr!(Pr, t)
        mul!(iRrBr, iRr(t), Br')
        mul!(Kr, iRrBr, Pr)
        # mul!(Kr, Rr(t)\Br', Pr)
        # mul!(Kr, Rr(t)\Br', Pr(t))
        return Kr
    end
    return _Kr
end

function riccati!(dP, P, (α,μ,fx!,fu!,Ar,Br,Rr,Qr,Kr), t)
    fx!(Ar, α(t), μ(t))
    fu!(Br, α(t), μ(t))
    # Kr = Rr(t)\Br'*P
    mul!(Kr, Rr(t)\Br', P)
    dP .= -Ar'P - P*Ar + Kr'*Rr(t)*Kr - Qr(t)
    # dP .= -Ar'P - P*Ar + Kr'*Br'*P - Qr(t)
end
