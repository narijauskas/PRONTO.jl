
# --------------------------------- regulator  η -> Kr --------------------------------- #

#TODO: regulator(model,t,α,μ)
# function regulator(::Val{NX},::Val{NU},α,μ,model) where {NX,NU}
function regulator(α,μ,model)
    NX = model.NX; NU = model.NU; 
    T = model.T;
    Qr = model.Qr; Rr = model.Rr; iRr = model.iRr;

    fx! = model.fx!
    _Ar = Buffer{Tuple{NX,NX}}()
    Ar(α,μ) = (fx!(_Ar,α,μ); return _Ar)
    Ar(t) = Ar(α(t),μ(t))
    # Ar(t) = (fx!(_Ar,α(t),μ(t)); return _Ar)
    # Ar(α,μ,t) = (fx!(_Ar,α(t),μ(t),t); return _Ar)

    fu! = model.fu!
    _Br = Buffer{Tuple{NX,NU}}()
    Br(α,μ) = (fu!(_Br,α,μ); return _Br)
    Br(t) = Br(α(t),μ(t))
    
    PT = collect(I(NX))
    Pr! = solve(ODEProblem(riccati!, PT, (T,0.0), (Ar,Br,Rr,Qr)))
    _Pr = Buffer{Tuple{NX,NX}}()
    Pr(t) = (Pr!(_Pr, t); return _Pr)
    
    _Kr = Buffer{Tuple{NU,NX}}()
    _iRrBr = Buffer{Tuple{NU,NX}}()
    #MAYBE: Kr(α,μ)
    function Kr(t)
        mul!(_iRrBr, iRr(t), Br(t)')
        mul!(_Kr, _iRrBr, Pr(t))
        return _Kr
    end
    return Kr
end

function riccati!(dP, P, (Ar,Br,Rr,Qr), t)
    # mul!(Kr, Rr(t)\Br(t)', P)
    Kr = Rr(t)\Br(t)'*P
    dP .= -Ar(t)'*P - P*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
    #NOTE: dP is symmetric, as should be P
end

# NOTE: is it possible to store Kr directly?