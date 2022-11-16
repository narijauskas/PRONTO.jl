using MacroTools
using MacroTools: postwalk

ex = begin
    NX = 3; NU = 2; NΘ = 1
end


function getNX(ex)
    postwalk(x->(@capture(x,NX=nx_); x), ex)
    return nx
end


1 == getNX(ex)


macro model(name, ex)
    nx = getNX(ex)
    quote
        struct $name <: PRONTO.Model{$nx,$nx,$nx}
        end

        let
            $ex
        end
    end
end

@macroexpand @model Foo begin
    NX = 3; NU = 2; NΘ = 1
end