export RVPosterior, compute_bic, compute_aicc

struct RVPosterior
    likes::Dict{String, RVLikelihood}
end

"""
    Construct an empty RVPosterior object 
"""
RVPosterior() = RVPosterior(Dict{String, RVLikelihood}())

Base.length(post::RVPosterior) = length(post.likes)
Base.merge!(post::RVPosterior, post2::RVPosterior) = merge!(post.likes, post2.likes)
Base.getindex(post::RVPosterior, key::String) = getindex(post.likes, key)
Base.firstindex(post::RVPosterior) = firstindex(post.likes)
Base.lastindex(post::RVPosterior) = lastindex(post.likes)
Base.iterate(post::RVPosterior) = iterate(post.likes)
Base.keys(post::RVPosterior) = keys(post.likes)
Base.values(post::RVPosterior) = values(post.likes)

"""
    compute_prior_logprob(post::RVPosterior, pars::Parameters)
Computes the cumulative natural logarithm of prior probability.
"""
compute_prior_logprob(post::RVPosterior, pars::Parameters) = compute_prior_logprob(pars)

"""
    compute_logaprob(post::RVPosterior, pars::Parameters)
Computes the a posteriori probability (lnL with explicit parameter prior knowledge).
"""
function compute_logaprob(post::RVPosterior, pars::Parameters)
    lnL = compute_prior_logprob(post, pars)
    if !isfinite(lnL)
        return -Inf
    end
    lnL += compute_logL(post, pars)
    if !isfinite(lnL)
        return -Inf
    end 
    return lnL
end

"""
    compute_logL(post::RVPosterior, pars::Parameters)
"""
function compute_logL(post::RVPosterior, pars::Parameters)
    lnL = 0.0
    for like in values(post)
        lnL += compute_logL(like, pars)
        if !isfinite(lnL)
            return -Inf
        end
    end
    return lnL
end


"""
    compute_redχ2(post::RVPosterior, pars::Parameters)
"""
function compute_redχ2(post::RVPosterior, pars::Parameters)
    χ2 = 0.0
    ν = 0
    for like ∈ values(post)
        data_t = get_times(like.data)
        residuals = compute_residuals(like, pars)
        errors = compute_data_errors(like, pars)
        if !isnothing(like.gp)
            noise_components = compute_noise_components(like, pars, data_t)
            for comp ∈ keys(noise_components)
                residuals[noise_components[comp][3]] .-= noise_components[comp][1][noise_components[comp][3]]
            end
        end
        χ2 += nansum((residuals ./ errors).^2)
        ν += length(residuals)
    end
    ν -= num_varied(pars)
    redχ2 = χ2 / ν
    return redχ2
end

num_data_points(post::RVPosterior) = sum([length(get_times(like.data)) for like ∈ values(post)])
      
"""
    compute_bic(post::RVPosterior, pars::Parameters)
Computes the Bayesian information criterion (BIC).
"""
function compute_bic(post::RVPosterior, pars::Parameters)
    n = num_data_points(post)
    k = num_varied(pars)
    lnL = compute_logL(post, pars)
    bic = k * log(n) - 2.0 * lnL
    return bic
end

"""
    compute_aicc(post::RVPosterior, pars::Parameters)
Computes the small-sample Akaike information criterion (AICc).
"""
function compute_aicc(post::RVPosterior, pars::Parameters)
    n = num_data_points(post)
    k = num_varied(pars)
    lnL = compute_logL(post, pars)
    aic = 2.0 * (k - lnL)
    d = n - k - 1
    if d > 0
        aicc = aic + (2 * k^2 + 2 * k) / d
    else
        aicc = Inf
    end
    return aicc
end