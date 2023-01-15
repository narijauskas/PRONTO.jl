using Pkg
Pkg.activate(".")

using Documenter
# using PRONTO

makedocs(
    sitename="PRONTO.jl",
    # modules = PRONTO
    authors = "Mantas Naris",
    pages = Any[
            "index.md"
        ],
    doctest=false,
    clean=true,
)

#= ----------- uncomment for push to main & deploy ----------- #

deploydocs(
    repo = "github.com/lab-collab/lab-collab.github.io",
    versions = nothing,
    # devbranch = "main",
    # devurl = ".",
)

# ----------------------------------------------------------- =#