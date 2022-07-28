using LinearAlgebra

export GaussianProcess, predict

"""
    Wrapper for a Gaussian Process noise kernel.
"""
struct GaussianProcess{K<:NoiseKernel}
    kernel::K
end

"""
    Make predictions with the GP model at arbitrary observation times.
"""
function predict end

"""
    compute_cov_matrix(gp::GaussianProcess{Kernel}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing) where{Kernel}
"""
function compute_cov_matrix(gp::GaussianProcess{Kernel}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing) where{Kernel}
    K = compute_cov_matrix(gp.kernel, pars, t1, t2)
    if !isnothing(data_rverr)
        n1, n2 = size(K)
        @assert n1 == n2 == length(data_rverr)
        dind = diagind(K)
        K[dind] .= diag(K) .+ data_rverr.^2
    end
    return K
end

"""
    predict(gp::GaussianProcess{Kernel}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing) where {Kernel}
"""
function predict(gp::GaussianProcess{Kernel}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing) where {Kernel}

    # Get grids
    if isnothing(tpred)
        tpred = data_t
    end

    # Get K
    K = compute_cov_matrix(gp, pars, data_t, data_t, data_rverr)
    
    # Compute version of K without intrinsic data error
    Ks = compute_cov_matrix(gp, pars, tpred, data_t, nothing)

    # Mean
    Chol = cholesky(K)
    α = Chol \ linpred
    μ = Ks * α

    # Error
    Kss = compute_cov_matrix(gp, pars, tpred, tpred, nothing)
    B = Chol \ collect(transpose(Ks))
    σ = sqrt.(diag(Kss .- Ks * B))

    return μ, σ
end