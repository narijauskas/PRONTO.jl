
# --------------------------------- regulator --------------------------------- #

# for regulator
function riccati!(dP, P, (Ar,Br,Rr,Qr,Kr), t)
    #TEST: hopefully optimized by compiler?
    # if not, do each step inplace to local buffers or SVectors
    dP .= -Ar(t)'P - P*Ar(t) + Kr(P,t)'*Rr(t)*Kr(P,t) - Qr(t)
end



# solve for regulator
# ξ or φ -> Kr
function regulator(α,μ,model)
    @unpack model
    T = last(ts)
    # ts = model.ts; T = last(ts); NX = model.NX; NU = model.NU

    Ar = Functor(NX,NX) do buf,t
        fx!(buf, α(t), μ(t))
    end

    Br = Functor(NX,NU) do buf,t
        fu!(buf, α(t), μ(t))
    end

    iRrBr = Functor(NU,NX) do buf,t
        mul!(buf, iRr(t), Br(t)')
    end

    # Kr = inv(Rr)*Br'*P
    Kr = Functor(NU,NX) do buf,P,t
        mul!(buf, iRrBr(t), P)
    end

    # PT,_ = arec(Ar(T), Br(T)*iRr(T)*Br(T)', Qr(T))
    PT = collect(I(NX))
    Pr = solve(ODEProblem(riccati!, PT, (T,0.0), (Ar,Br,Rr,Qr,Kr)))

    Kr = Functor(NU,NX) do buf,t
        mul!(buf, iRrBr(t), Pr(t))
    end

    return Kr
end



function Regulator(α,μ,NX,NU,fx!,fu!)
    
    Ar = Functor(NX,NX) do buf,t
        fx!(buf, α(t), μ(t))
    end
    # Ar = MMatrix{NX,NX,Float64}(undef)
    # fx!(Ar, α(t), μ(t))
    function Kr(t)
    end
end


# function regulator(...)
#     ...
#     function Kr(t)
#     end
# end

# Kr = regulator(...)