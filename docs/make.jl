using GridapSolvers
using Documenter

DocMeta.setdocmeta!(GridapSolvers, :DocTestSetup, :(using GridapSolvers); recursive=true)

makedocs(;
    modules=[GridapSolvers],
    authors="Santiago Badia <santiago.badia@monash.edu>, Jordi Manyer <jordi.manyer@monash.edu>, Alberto F. Martin <alberto.martin@monash.edu>",
    repo="https://github.com/gridap/GridapSolvers.jl/blob/{commit}{path}#{line}",
    sitename="GridapSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gridap.github.io/GridapSolvers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gridap/GridapSolvers.jl",
    devbranch="main",
)
