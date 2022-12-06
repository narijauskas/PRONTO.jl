# ----------------------------------- * runtime feedback ----------------------------------- #
using Crayons

as_tag(str) = as_tag(crayon"default", str)
as_tag(c::Crayon, str) = as_color(c, as_bold("[$str: "))
as_color(c::Crayon, str) = "$c" * str * "$(crayon"default")"
as_bold(ex) = as_bold(string(ex))
as_bold(str::String) = "$(crayon"bold")" * str * "$(crayon"!bold")"
clearln() = print("\e[2K","\e[1G")

info(str) = println(as_tag(crayon"magenta","PRONTO"), str)
info(i, str) = println(as_tag(crayon"magenta","PRONTO[$i]"), str)
iinfo(str) = println("    > ", str) # secondary-level
iiinfo(str) = println("        > ", str) 



# ----------------------------------- * code timing ----------------------------------- #

tick(name) = esc(Symbol(String(name)*"_tick"))
tock(name) = esc(Symbol(String(name)*"_tock"))

macro tick(name=:(_))

    :($(tick(name)) = time_ns())
end

macro tock(name=:(_))

    :($(tock(name)) = time_ns())
end

macro clock(name=:(_))

    _tick = tick(name)
    _tock = tock(name)
    ms = :(($_tock - $_tick)/1e6)
    :("$($:(round($ms; digits=3))) ms")
end


# barplot(["dPr/dt","dx/dt"],rand(2), width=72, color=:magenta,title="Runtime Stats")