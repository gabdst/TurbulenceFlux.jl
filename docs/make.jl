push!(LOAD_PATH, "../")
using Documenter
using TurbulenceFlux

makedocs(
    sitename = "TurbulenceFlux",
    format = Documenter.HTML(),
    modules = [TurbulenceFlux],
    remotes = nothing,
)
