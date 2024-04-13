export ChromaticKernelJ3

struct ChromaticKernelJ3 <: NoiseKernel
    par_names::Vector{String}
end
 
function compute_cov_matrix(kernel::ChromaticKernelJ3, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real})

    par_names = kernel.par_names

    # Distance matrix
    dist_matrix = compute_stationary_dist_matrix(t1, t2)
    λ_matrix = compute_stationary_dist_matrix(λvec1, λvec2)
    
    # Alias params
    gp_amp = pars[par_names[1]].value
    exp_length_λ = pars[par_names[2]].value
    exp_length_t = pars[par_names[3]].value
    per_length = pars[par_names[4]].value
    per = pars[par_names[5]].value

    # Construct terms
    decay_term_λ = @. -0.5 * (λ_matrix / exp_length)^2
    decay_term_t = @. -0.5 * (dist_matrix / exp_length_t)^2
    periodic_term = @. -0.5 * (1 / per_length)^2 * sin(π * dist_matrix / per)^2
    
    # Construct full cov matrix
    K = @. gp_amp^2 * exp(decay_term_λ + decay_term_t + periodic_term)
    
    return K
end


function compute_cov_matrix(gp::GaussianProcess{ChromaticKernelJ3}, pars::Parameters, t1::AbstractVector{<:Real}, t2::AbstractVector{<:Real}, λvec1::AbstractVector{<:Real}, λvec2::AbstractVector{<:Real}, data_rverr::Union{AbstractVector{<:Real}, Nothing}=nothing)
    K = compute_cov_matrix(gp.kernel, pars, t1, t2, λvec1, λvec2)
    if !isnothing(data_rverr)
        n1, n2 = size(K)
        @assert n1 == n2 == length(data_rverr)
        dind = diagind(K)
        K[dind] .= diag(K) .+ data_rverr.^2
    end
    return K
end


function predict(gp::GaussianProcess{ChromaticKernelJ3}, pars::Parameters, data_t::AbstractVector{<:Real}, linpred::AbstractVector{<:Real}, data_rverr::AbstractVector{<:Real}, λs::AbstractVector{<:Real}; tpred::Union{AbstractVector{<:Real}, Nothing}=nothing, λpred::Real)

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