
# --------------------------------- regulator --------------------------------- #

function regulator(NX,NU,T,α,μ,fx!,fu!,Rr,Qr)
    Ar = Buffer(NX,NX) 
    Br = Buffer(NX,NU)
    PT = collect(I(NX))
    Pr = solve(ODEProblem(riccati!, PT, (T,0.0), (α,μ,fx!,fu!,Ar,Br,Rr,Qr)))
    buf = Buffer(NU,NX)
    function Kr(t)
        fu!(Br, α(t), μ(t))
        mul!(buf, Rr(t)\Br', Pr(t))
        return buf
    end
    return Kr
end

function riccati!(dP, P, (α,μ,fx!,fu!,Ar,Br,Rr,Qr), t)
    fx!(Ar, α(t), μ(t))
    fu!(Br, α(t), μ(t))
    Kr = Rr(t)\Br'*P
    dP .= -Ar'P - P*Ar + Kr'*Rr(t)*Kr - Qr(t)
    # dP .= -Ar'P - P*Ar + Kr'*Br'*P - Qr(t)
end
