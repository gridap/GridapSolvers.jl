using GridapSolvers
using Documenter

DocMeta.setdocmeta!(GridapSolvers, :DocTestSetup, :(using GridapSolvers); recursive=true)

makedocs(;
    modules=[GridapSolvers,GridapSolvers.BlockSolvers],
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
        "SolverInterfaces" => "SolverInterfaces.md",
        "MultilevelTools" => "MultilevelTools.md",
        "LinearSolvers" => "LinearSolvers.md",
        "NonlinearSolvers" => "NonlinearSolvers.md",
        "BlockSolvers" => "BlockSolvers.md",
        "PatchBasedSmoothers" => "PatchBasedSmoothers.md",
    ],
)

deploydocs(;
    repo="github.com/gridap/GridapSolvers.jl",
    devbranch="main",
)
