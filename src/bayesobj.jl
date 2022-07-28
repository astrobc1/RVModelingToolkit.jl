using LinearAlgebra

export compute_logL, compute_logaprob, compute_cov_matrix, compute_residuals, compute_data_errors, compute_noise_components, compute_redχ2

const TWO_PI = 2 * π
const LOG_2PI = log(TWO_PI)

"""
    compute_redχ2(residuals, errors, ν)
Utility to compute the reduced chi-squared metric.
"""
function compute_redχ2(residuals::AbstractArray{<:Real}, errors::AbstractArray{<:Real}, ν::Real)
    return nansum((residuals ./ errors).^2) / ν
end

"""
    compute_logL
Method to compute the logarithm of the a likelihood. Implemented by the likelihood and posterior objects.
"""
function compute_logL end

"""
    compute_logaprob
Method to compute the logarithm of the a posteriori probability. Implemented by the posterior object.
"""
function compute_logaprob end

"""
    compute_cov_matrix
Method to compute the covariance matrix. Must be implemented for a noise kernel, GP, and likelihood object.
"""
function compute_cov_matrix end

"""
    compute_residuals
Method to compute the residuals, still including correlated noise. Implemented by each likelihood.
"""
function compute_residuals end

"""
    compute_data_errors
Method to compute the data errors. Implemented by each likelihood.
"""
function compute_data_errors end

"""
    compute_noise_components
Method to compute the noise components. Implemented by each likelihood / kernel.
"""
function compute_noise_components end

include("likelihood.jl")
include("posterior.jl")