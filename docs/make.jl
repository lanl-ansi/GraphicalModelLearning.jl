using Documenter, GraphicalModelLearning

makedocs(
    modules = [GraphicalModelLearning],
    format = Documenter.HTML(),
    sitename = "GraphicalModelLearning",
    authors = "Carleton Coffrin, Andrey Y. Lokhov, Sidhant Misra, Marc Vuffray, and contributors.",
    pages = [
        "Home" => "index.md",
        "Library" => "library.md"
    ]
)

deploydocs(
    repo = "github.com/lanl-ansi/GraphicalModelLearning.jl.git",
)
