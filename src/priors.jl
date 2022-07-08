module Priors

export Prior, logprob

using Infiltrator

abstract type Prior end
function logprob end
struct Gaussian <: Prior
    μ::Float64
    σ::Float64
end
function logprob(p::Gaussian, x)
    return -0.5 * ((x - p.μ) / p.σ)^2 - 0.5 * log((p.σ^2) * 2 * π)
end

function Base.show(io::IO, p::Gaussian)
    println(io, "Gaussian($(p.μ), $(p.σ))")
end


struct Uniform <: Prior
    lower_bound::Float64
    upper_bound::Float64
end

function logprob(p::Uniform, x)
    if p.lower_bound < x < p.upper_bound
        return -1 * log(p.upper_bound - p.lower_bound)
    else
       return -Inf
    end
end

function Base.show(io::IO, p::Uniform)
    println(io, "Uniform($(p.lower_bound), $(p.upper_bound))")
end

struct Positive <: Prior end
function logprob(p::Positive, x)
    if x > 0
        return 0
    else
        return -Inf
    end
end

function Base.show(io::IO, p::Positive)
    println(io, "Positive")
end

struct JeffreysGD <: Prior
    lower_bound::Float64
    upper_bound::Float64
    knee::Float64
    lognorm::Float64
end
JeffreysGD(lower_bound, upper_bound, knee=0.0) = JeffreysGD(lower_bound, upper_bound, knee, log(1.0 / log((upper_bound - knee) / (lower_bound - knee))))
function Base.show(io::IO, p::JeffreysGD)
    println(io, "Jeffrey's Gaussian ($(p.lower_bound), $(p.knee), $(p.upper_bound))")
end


function logprob(p::JeffreysGD, x)
    if p.lower_bound < x < p.upper_bound
        return p.lognorm - log(x - p.knee)
    else
        return -Inf
    end
end

end