using Infiltrator

export ChromaticKernelJ1, ChromaticKernelJ2

"""
Type for a ChromaticKernelJ1 GP kernel. `par_names` is a vector containing the names of the hyper-parameters, which must be in the following order, however can use arbitrary names.
    1. amplitude
    2. exponential length scale
    3. period length scale
    4. period
"""
struct ChromaticKernelJ1 <: NoiseKernel
    par_names::Vector{String}
end

"""
    compute_cov_matrix(kernel::ChromaticKernelJ1, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, amp_vec1::AbstractVector{<:Real}, amp_vec2::AbstractVector{<:Real})
Computes the covariance matrix for the J1 chromatic kernel. This method also needs two corresponding vectors contaning the amplitudes of each observation time (static for a given spectrograph).
"""
function compute_cov_matrix(kernel::ChromaticKernelJ1, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, amp_vec1::AbstractVector{<:Real}, amp_vec2::AbstractVector{<:Real})
    par_names = kernel.par_names
    n_instruments = length([pname for pname ∈ par_names if startswith(pname, "gp_amp")])
    dist_matrix = compute_stationary_dist_matrix(t1, t2)
    amp_matrix = gen_amp_matrix(kernel, amp_vec1, amp_vec2)
    exp_length = pars[par_names[n_instruments+1]].value
    per_length = pars[par_names[n_instruments+2]].value
    per = pars[par_names[n_instruments+3]].value
    decay_term = @. -0.5 * (dist_matrix / exp_length)^2
    periodic_term = @. -0.5 * (1 / per_length)^2 * sin(π * dist_matrix / per)^2
    K = @. amp_matrix^2 * exp(decay_term + periodic_term)
    return K
end

function gen_amp_matrix(kernel::ChromaticKernelJ1, amp_vec1, amp_vec2)
    return sqrt.(amp_vec1 * amp_vec2')
end

"""
    compute_cov_matrix(gp::GaussianProcess{ChromaticKernelJ1}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, amp_vec1::AbstractVector{<:Real}, amp_vec2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing)
"""
function compute_cov_matrix(gp::GaussianProcess{ChromaticKernelJ1}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, amp_vec1::AbstractVector{<:Real}, amp_vec2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing)
    K = compute_cov_matrix(gp.kernel, pars, t1, t2, amp_vec1, amp_vec2)
    if !isnothing(data_rverr)
        n1, n2 = size(K)
        @assert n1 == n2 == length(data_rverr)
        dind = diagind(K)
        K[dind] .= diag(K) .+ data_rverr.^2
    end
    return K
end

"""
    predict(gp::GaussianProcess{ChromaticKernelJ1}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}, data_amp_vec::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing, amppred::AbstractVector{<:Real})
"""
function predict(gp::GaussianProcess{ChromaticKernelJ1}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}, data_amp_vec::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing, amppred::Real)

    # Get grids
    if isnothing(tpred)
        tpred = data_t
    end
    
    # Get K
    amp_vec_pred = fill(amppred, length(tpred))
    K = compute_cov_matrix(gp, pars, data_t, data_t, data_amp_vec, data_amp_vec, data_rverr)
    
    # Compute version of K without intrinsic data error
    Ks = compute_cov_matrix(gp, pars, tpred, data_t, amp_vec_pred, data_amp_vec, nothing)

    # Mean
    Chol = cholesky(K)
    α = Chol \ linpred
    μ = Ks * α

    # Error
    Kss = compute_cov_matrix(gp, pars, tpred, tpred, amp_vec_pred, amp_vec_pred, nothing)
    B = Chol \ collect(transpose(Ks))
    σ = sqrt.(diag(Kss .- Ks * B))

    return μ, σ
end


##################################################################################################
##################################################################################################
##################################################################################################


"""
Type for a ChromaticKernelJ2 GP kernel. `par_names` is a vector containing the names of the hyper-parameters, which must be in the following order, however can use arbitrary names.
    1. amplitude
    2. exponential length scale
    3. period length scale
    4. period
"""
struct ChromaticKernelJ2 <: NoiseKernel
    par_names::Vector{String}
    λ0::Float64
end
 
"""
    compute_cov_matrix(kernel::ChromaticKernelJ2, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real})
"""
function compute_cov_matrix(kernel::ChromaticKernelJ2, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real})

    par_names = kernel.par_names

    # Distance matrix
    dist_matrix = compute_stationary_dist_matrix(t1, t2)

    # Wave matrix
    wave_matrix = gen_λ_matrix(kernel, λvec1, λvec2)
    
    # Alias params
    gp_amp_0 = pars[par_names[1]].value
    gp_amp_scale = pars[par_names[2]].value
    exp_length = pars[par_names[3]].value
    per_length = pars[par_names[4]].value
    per = pars[par_names[5]].value

    # Construct QP terms
    decay_term = @. -0.5 * (dist_matrix / exp_length)^2
    periodic_term = @. -0.5 * (1 / per_length)^2 * sin(π * dist_matrix / per)^2
    
    # Construct full cov matrix
    K = @. gp_amp_0^2 * (kernel.λ0 / sqrt(wave_matrix))^(2 * gp_amp_scale) * exp(decay_term + periodic_term)
    
    return K
end

function gen_λ_matrix(kernel::ChromaticKernelJ2, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real})
    return λvec1 * λvec2'
end

"""
    compute_cov_matrix(gp::GaussianProcess{ChromaticKernelJ2}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing)
"""
function compute_cov_matrix(gp::GaussianProcess{ChromaticKernelJ2}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing)
    K = compute_cov_matrix(gp.kernel, pars, t1, t2, λvec1, λvec2)
    if !isnothing(data_rverr)
        n1, n2 = size(K)
        @assert n1 == n2 == length(data_rverr)
        dind = diagind(K)
        K[dind] .= diag(K) .+ data_rverr.^2
    end
    return K
end


"""
    predict(gp::GaussianProcess{ChromaticKernelJ2}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}, λs::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing, λpred::AbstractVector{<:Real})
"""
function predict(gp::GaussianProcess{ChromaticKernelJ2}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}, λs::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing, λpred::Real)

    # Get grids
    if isnothing(tpred)
        tpred = data_t
    end
    
    # Get K
    λpred_vec = fill(λpred, length(tpred))
    K = compute_cov_matrix(gp, pars, data_t, data_t, λs, λs, data_rverr)
    
    # Compute version of K without intrinsic data error
    Ks = compute_cov_matrix(gp, pars, tpred, data_t, λpred_vec, λs, nothing)

    # Mean
    Chol = cholesky(K)
    α = Chol \ linpred
    μ = Ks * α

    # Error
    Kss = compute_cov_matrix(gp, pars, tpred, tpred, λpred_vec, λpred_vec, nothing)
    B = Chol \ collect(transpose(Ks))
    σ = sqrt.(diag(Kss .- Ks * B))


    return μ, σ
end