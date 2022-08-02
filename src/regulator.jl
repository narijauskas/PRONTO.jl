
# --------------------------------- regulator  η -> Kr --------------------------------- #

#TODO: regulator(model,t,α,μ)
# function regulator(::Val{NX},::Val{NU},α,μ,model) where {NX,NU}
function regulator(α,μ,model)
    NX = model.NX; NU = model.NU; 
    T = model.T;
    Qr = model.Qr; Rr = model.Rr; iRr = model.iRr;

    fx! = model.fx!
    _Ar = Buffer{Tuple{NX,NX}}()
    Ar(t) = (fx!(_Ar,α(t),μ(t)); return _Ar)
    Ar(x,u) = (fx!(_Ar,x,u); return _Ar)
    Ar(x,u,t) = (fx!(_Ar,x(t),u(t),t); return _Ar)

    fu! = model.fu!; _Br = Buffer{Tuple{NX,NU}}()
    Br(t) = (fu!(_Br,α(t),μ(t)); return _Br)
    
    _Kr = Buffer{Tuple{NU,NX}}()
    
    PT = collect(I(NX))
    Pr! = solve(ODEProblem(riccati!, PT, (T,0.0), (Ar,Br,Rr,Qr,_Kr)))
    _Pr = Buffer{Tuple{NX,NX}}()
    Pr = @closure (t)->(Pr!(_Pr, t); return _Pr)
    # Pr = functor((Pr,t) -> Pr!(Pr, t), Buffer{Tuple{NX,NX}}())
    
    iRrBr = Buffer{Tuple{NU,NX}}()
    function Kr(t)
        mul!(iRrBr, iRr(t), Br(t)')
        mul!(_Kr, iRrBr, Pr(t))
        return _Kr
    end
    return Kr
end

function riccati!(dP, P, (Ar,Br,Rr,Qr,Kr), t)
    # mul!(Kr, Rr(t)\Br(t)', P)
    Kr = Rr(t)\Br(t)'*P
    dP .= -Ar(t)'*P - P*Ar(t) + Kr'*Rr(t)*Kr - Qr(t)
    #NOTE: dP is symmetric, as should be P
end

# NOTE: is it possible to store Kr directly?