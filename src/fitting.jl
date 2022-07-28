using IterativeNelderMead

export run_mapfit, run_mcmc, model_comparison, planet_stats

using Infiltrator

include("mapfit.jl")
include("mcmc.jl")