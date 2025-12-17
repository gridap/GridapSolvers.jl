using Documenter

using GridapSolvers
using GridapPETSc, GridapP4est, IterativeSolvers, Pardiso

# Examples
include("examples.jl")

# Changelog
cp(string(@__DIR__,"/../NEWS.md"), string(@__DIR__,"/src/changelog.md"))

# Extensions
extensions = map(
    ext -> Base.get_extension(GridapSolvers,ext),
    (:GridapP4estExt,:GridapPETScExt,:IterativeSolversExt,:PardisoExt)
)
println(" >>> Extensions found: ", extensions)

# Make Docs
DocMeta.setdocmeta!(GridapSolvers, :DocTestSetup, :(using GridapSolvers); recursive=true)

makedocs(;
    modules = [GridapSolvers,extensions...],
    authors = "Santiago Badia <santiago.badia@monash.edu>, Jordi Manyer <jordi.manyer@monash.edu>, Alberto F. Martin <alberto.martin@monash.edu>",
    repo = "https://github.com/gridap/GridapSolvers.jl/blob/{commit}{path}#{line}",
    sitename = "GridapSolvers.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://gridap.github.io/GridapSolvers.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Modules" => [
            "SolverInterfaces" => "SolverInterfaces.md",
            "MultilevelTools" => "MultilevelTools.md",
            "LinearSolvers" => "LinearSolvers.md",
            "NonlinearSolvers" => "NonlinearSolvers.md",
            "BlockSolvers" => "BlockSolvers.md",
            "PatchBasedSmoothers" => "PatchBasedSmoothers.md",
        ],
        "Extensions" => [
            "GridapP4est.jl" => "Extensions/GridapP4est.md",
            "GridapPETSc.jl" => "Extensions/GridapPETSc.md",
            "IterativeSolvers.jl" => "Extensions/IterativeSolvers.md",
            "Pardiso.jl" => "Extensions/Pardiso.md",
        ],
        "Examples" => [
            "Stokes" => "Examples/Stokes.md",
            "Navier-Stokes" => "Examples/NavierStokes.md",
            "Stokes (GMG)" => "Examples/StokesGMG.md",
            "Navier-Stokes (GMG)" => "Examples/NavierStokesGMG.md",
            "Darcy (GMG)" => "Examples/DarcyGMG.md",
        ],
        "Changelog" => "changelog.md",
    ],
    warnonly = [:doctest,:example_block,:eval_block],
    clean = true,
)

deploydocs(;
    repo="github.com/gridap/GridapSolvers.jl",
    devbranch="main",
)
