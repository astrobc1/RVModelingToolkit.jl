module Priors

export Prior, logprob

"""
    Base types for all priors.
"""
abstract type Prior end

"""
    Computes the natural logarithm of the probability for this prior given a parameter value.
"""
function logprob end

"""
    A Gaussian prior with mean μ::Float64 and standard deviation σ::Float64.
"""
struct Gaussian <: Prior
    μ::Float64
    σ::Float64
end

"""
    logprob(p::Gaussian, x)
"""
function logprob(p::Gaussian, x)
    return -0.5 * ((x - p.μ) / p.σ)^2 - 0.5 * log((p.σ^2) * 2 * π)
end

function Base.show(io::IO, p::Gaussian)
    println(io, "Gaussian($(p.μ), $(p.σ))")
end

"""
    A uniform prior with finite bounds.
"""
struct Uniform <: Prior
    lower_bound::Float64
    upper_bound::Float64
end

"""
    logprob(p::Uniform, x)
"""
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

"""
    A trait for a positive (improper) prior to keep a parameter > 0, useful for semi-amplitudes.
"""
struct Positive <: Prior end

"""
    logprob(p::Positive, x)
"""
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

"""
    JeffreysGD(lower_bound::Real, upper_bound::Real, knee::Real=0.0)
Construct a JeffreysGD (~ (x-x0)^-1) bounded by `lower_bound` and `upper_bound` with a knee `x0`.
"""
JeffreysGD(lower_bound::Real, upper_bound::Real, knee::Real=0.0) = JeffreysGD(lower_bound, upper_bound, knee, log(1.0 / log((upper_bound - knee) / (lower_bound - knee))))

function Base.show(io::IO, p::JeffreysGD)
    println(io, "Jeffrey's Gaussian ($(p.lower_bound), $(p.knee), $(p.upper_bound))")
end

"""
    logprob(p::JeffreysGD, x)
"""
function logprob(p::JeffreysGD, x)
    if p.lower_bound < x < p.upper_bound
        return p.lognorm - log(x - p.knee)
    else
        return -Inf
    end
end

end