using Documenter, GraphicalModelLearning

makedocs(
    modules = [GraphicalModelLearning],
    format = :html,
    sitename = "GraphicalModelLearning",
    authors = "Carleton Coffrin, Andrey Y. Lokhov, Sidhant Misra, Marc Vuffray, and contributors.",
    pages = [
        "Home" => "index.md",
        "Library" => "library.md"
    ]
)

deploydocs(
    deps = nothing,
    make = nothing,
    target = "build",
    repo = "github.com/lanl-ansi/GraphicalModelLearning.jl.git",
    julia = "0.6"
)
