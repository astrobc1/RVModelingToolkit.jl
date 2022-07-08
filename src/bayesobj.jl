using Infiltrator, PyPlot
using LinearAlgebra

const TWO_PI = 2 * π
const LOG_2PI = log(TWO_PI)

function redχ2loss(residuals, errors, ν)
    return nansum((residuals ./ errors).^2) / ν
end

include("likelihood.jl")
include("posterior.jl")