# use with caution
macro unpack(model)
    return esc(quote
        NX = $(model).NX
        NU = $(model).NU
        T = $(model).T
        ts = $(model).ts
        # f = $(model).f
        fx! = $(model).fx!
        fu! = $(model).fu!
        fxx! = $(model).fxx!
        fxu! = $(model).fxu!
        fuu! = $(model).fuu!
        l = $(model).l
        lx! = $(model).lx!
        lu! = $(model).lu!
        lxx! = $(model).lxx!
        lxu! = $(model).lxu!
        luu! = $(model).luu!
        p = $(model).p
        px! = $(model).px!
        pxx! = $(model).pxx!
        x0 = $(model).x0
        Qr = $(model).Qr
        Rr = $(model).Rr
        iRr = $(model).iRr
    end)
end