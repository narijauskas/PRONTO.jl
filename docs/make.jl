using Documenter
# using PRONTO

makedocs(
    sitename = "PRONTO.jl",
    # modules = PRONTO
    authors = "Mantas Naris and Jay Shao",
    pages = Any[
            "index.md"
        ],
    doctest = false,
    clean = true,
)


deploydocs(
    repo = "github.com/narijauskas/PRONTO.jl.git",
    push_preview = true,
)

