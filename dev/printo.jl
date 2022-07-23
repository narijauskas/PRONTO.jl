# https://discourse.julialang.org/t/how-clear-the-printed-content-in-terminal-and-print-to-the-same-line/19549/6
print("ayy")
sleep(0.5)
print("\e[2K") # clear whole line
print("\e[1G") # move cursor to column 1
print("lmao")

# print over the previous line
printo(str) = print("\e[2K","\e[1G", str)

print("ayy")
sleep(0.5)
printo("lmao")


using Crayons
as_bold(str) = "$(crayon"bold")" * str * "$(crayon"!bold")"
as_color(c::Crayon, str) = "$c" * str * "$(crayon"default")"
as_tag(c::Crayon, str) = as_color(c, as_bold("[$str: "))
as_tag(str) = as_tag(crayon"default", str)

print(as_tag("PRONTO"))
sleep(0.5)
printo(as_tag(crayon"magenta","PRONTO"))


# pronto info
info(str) = print("\e[2K","\e[1G", as_tag(crayon"magenta","PRONTO"), str)
# important info
# iinfo(str) = print("\e[2K","\e[1G", as_tag("PRONTO"), str, "\n")


prontoinfo("ayy")
sleep(0.5)
prontoinfo("lmao")