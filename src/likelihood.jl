
export RVLikelihood

mutable struct RVLikelihood{M, N}
    data::CompositeRVData
    model::M
    gp::N
    RVLikelihood(data, model, gp=nothing) = new{typeof(model), typeof(gp)}(data, model, gp)
end

"""
    compute_logL(like::RVLikelihood{<:Any, <:Nothing}, pars::parameters)
Computes ln(L) for this likelihood.
"""
function compute_logL(like::RVLikelihood{<:Any, <:Nothing}, pars::Parameters)
    residuals = compute_residuals(like, pars)
    errors = compute_data_errors(like, pars)
    n = length(residuals)
    lnL = -0.5 * (sum((residuals ./ errors).^2) + sum(log.(errors.^2)) + n * LOG_2PI)
    return lnL
end

function compute_logL(like::RVLikelihood{<:Any, <:GaussianProcess{Kernel}}, pars::Parameters) where {Kernel}
    residuals = compute_residuals(like, pars)
    errors = compute_data_errors(like, pars)
    n = length(residuals)
    try
        K = compute_cov_matrix(like, pars)
        Chol = cholesky(K)
        α = Chol \ residuals
        lndetK = logdet(K)
        lnL = -0.5 * ((transpose(residuals) * α) + lndetK + n * LOG_2PI)
    catch
        return -Inf
    end
end


function compute_cov_matrix(like::RVLikelihood{<:Any, <:GaussianProcess{Kernel}}, pars) where {Kernel}
    data_rverr = compute_data_errors(like, pars)
    data_t = get_times(like.data)
    K = compute_cov_matrix(like.gp, pars, data_t, data_t, data_rverr)
    return K
end

function compute_cov_matrix(like::RVLikelihood{<:Any, <:GaussianProcess{ChromaticKernelJ1}}, pars)
    data_rverr = compute_data_errors(like, pars)
    data_t = get_times(like.data)
    data_amp_vec = fill(NaN, length(data_rverr))
    data_instname_vec = get_instnames(like.data)
    for i=1:length(data_instname_vec)
        instname = data_instname_vec[i]
        data_amp_vec[i] = pars["gp_amp_$(instname)"].value
    end
    K = compute_cov_matrix(like.gp, pars, data_t, data_t, data_amp_vec, data_amp_vec, data_rverr)
    return K
end

function compute_cov_matrix(like::RVLikelihood{<:Any, <:GaussianProcess{ChromaticKernelJ2}}, pars)
    data_rverr = compute_data_errors(like, pars)
    data_t = get_times(like.data)
    data_λ_vec = get_λs(like.data)
    K = compute_cov_matrix(like.gp, pars, data_t, data_t, data_λ_vec, data_λ_vec, data_rverr)
    return K
end


function compute_residuals(like::RVLikelihood, pars)
    data_t = get_times(like.data)
    data_rv = get_rvs(like.data)
    model_rv = build(like.model, pars, data_t)
    residuals = data_rv .- model_rv .- build_trend_zero(like.model, like.data, pars, data_t)
    return residuals
end

function compute_data_errors(like::RVLikelihood, pars)
    errors2 = get_rverrs(like.data).^2
    for instname ∈ keys(like.data)
        inds = like.data.indices[instname]
        errors2[inds] .+= pars["jitter_$(instname)"].value^2
    end
    return sqrt.(errors2)
end


function compute_noise_components(like::RVLikelihood{<:Any, <:GaussianProcess{Kernel}}, pars, t) where {Kernel}
    residuals = compute_residuals(like, pars)
    data_t = get_times(like.data)
    data_rverr = compute_data_errors(like, pars)
    noise_components = Dict{String, Any}()
    label = "GP ["
    for key ∈ keys(like.data)
        label *= "$key,"
    end
    label = label[1:end-1]
    label *= "]"
    inds = 1:length(data_t)
    gp, gperr = predict(like.gp, pars, data_t, residuals, data_rverr, tpred=t)
    noise_components[label] = (gp, gperr, inds)
    return noise_components
end


function compute_noise_components(like::RVLikelihood{<:Any, <:GaussianProcess{ChromaticKernelJ1}}, pars, t)
    residuals = compute_residuals(like, pars)
    data_t = get_times(like.data)
    data_rverr = compute_data_errors(like, pars)
    noise_components = Dict{String, Any}()
    data_amp_vec = fill(NaN, length(data_rverr))
    data_instname_vec = get_instnames(like.data)
    for i=1:length(data_instname_vec)
        instname = data_instname_vec[i]
        data_amp_vec[i] = pars["gp_amp_$(instname)"].value
    end
    for instname ∈ keys(like.data)
        label = "GP $(instname)"
        amp = pars["gp_amp_$(instname)"].value
        gp, gperr = predict(like.gp, pars, data_t, residuals, data_rverr, data_amp_vec, tpred=t, amppred=amp)
        inds = findall(data_instname_vec .== instname)
        noise_components[label] = (gp, gperr, inds)
    end
    return noise_components
end


function compute_noise_components(like::RVLikelihood{<:Any, <:GaussianProcess{ChromaticKernelJ2}}, pars, t)
    residuals = compute_residuals(like, pars)
    data_t = get_times(like.data)
    data_rverr = compute_data_errors(like, pars)
    noise_components = Dict{String, Any}()
    λs = get_λs(like.data)
    λs_unq = unique(λs)
    for λ in λs_unq
        label = "GP $(λ)"
        gp, gperr = predict(like.gp, pars, data_t, residuals, data_rverr, λs, tpred=t, λpred=λ)
        inds = findall(λs .== λ)
        noise_components[label] = (gp, gperr, inds)
    end
    return noise_components
end

