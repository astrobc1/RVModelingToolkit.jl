pushfirst!(LOAD_PATH,"../src/")

using Documenter
using RVModelingToolkit

makedocs(
    sitename = "RVModelingToolkit",
    format = Documenter.HTML(),
    modules = [RVModelingToolkit, Priors],
    pages = [
        "index.md",
        "examples.md",
        "api.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/astrobc1/RVModelingToolkit.jl.git"
)
