using AffineInvariantMCMC
using Statistics, NaNStatistics

function run_mcmc(post::RVPosterior, p0::Parameters; n_burn_steps, τrel_thresh=0.01, n_min_steps=1000, n_τs_thresh=40, check_every=100, n_max_steps=100_000)

    vecs = to_vecs(p0)
    vi = findall(vecs.vary)
    n_pars_vary = length(vi)
    pnamesv = collect(keys(p0))[vi]
    ptest = deepcopy(p0)
    pbest = deepcopy(p0)
    pmed = deepcopy(p0)

    obj_wrapper = (x) -> begin
        for i ∈ eachindex(x)
            ptest[pnamesv[i]].value = x[i]
        end
        return compute_logaprob(post, ptest)
    end

    mcmc_result = (;pbest=pbest, lnLbest=NaN, lnLmed=NaN, n_steps=nothing, autocorrs=nothing, chains=nothing, lnLs=nothing, acc=nothing)

    walkers = get_initial_walkers(post, p0)
    n_walkers = size(walkers)[2]

    println("Running MCMC burn in ($(n_burn_steps) steps) ...")

    # Chains size = (n_walkers, n_pars, n_steps)
    # lnLs = (n_walkers, n_steps)
    chain, lnLs = AffineInvariantMCMC.sample(obj_wrapper, n_walkers, walkers, n_burn_steps, 1)
    
    println("Completed MCMC burn in")

    walkers = chain[:, :, end]
    
    println("Running MCMC ...")

    chains_full = fill(NaN, (0, n_walkers, n_pars_vary))
    lnLs_full = fill(NaN, (0, n_walkers))
    n_art_steps = Int(floor(2 * n_max_steps / check_every))

    τmeds = Float64[]
    τold = Inf
    line_old = ""
    ii = 0

    for i=1:n_art_steps

        ii = (i - 1) * check_every + 1

        # Sample
        chains, lnLs = AffineInvariantMCMC.sample(obj_wrapper, n_walkers, walkers, check_every, 1)
        walkers .= chains[:, :, end]
        chains_full = cat(chains_full, permutedims(chains, (3, 2, 1)), dims=1)
        lnLs_full = cat(lnLs_full, permutedims(lnLs, (2, 1)), dims=1)

        # Check convergence:
        # 1. Ensure we've run a sufficient number of autocorrelation time scales
        # 2. Ensure the estimations of the autorr times themselves are settling.
        τs = get_autocorr_times(chains_full)
        τmed = nanmedian(τs)
        push!(τmeds, τmed)
        τrel = abs(τold - τmed) / τmed
        line = " τ = $(round(τmed, digits=5)) / x $(n_τs_thresh), rel Δτ = $(round(τrel, digits=5)) / $(round(τrel_thresh, digits=5)), steps = $(n_min_steps) / $(ii) / $(n_max_steps)"
        print("\r" * repeat(" ", length(line_old)))
        print("\r τ = $(round(τmed, digits=5)) / x $(n_τs_thresh), rel Δτ = $(round(τrel, digits=5)) / $(round(τrel_thresh, digits=5)), steps = $(n_min_steps) / $(ii) / $(n_max_steps)")
        if ((τrel < τrel_thresh) && (ii > n_min_steps) && (ii > τmed * n_τs_thresh))
            println("\nConverged!")
            break
        end
        if ii > n_max_steps
            @warn "MCMC failed to converge!"
            break
        end
        τold = τmed
        line_old = line
    end

    println("Completed MCMC")

    n_steps = size(chains_full)[1]
    flat_chains = reshape(chains_full, (n_walkers * n_steps, n_pars_vary))
    flat_lnLs = reshape(lnLs_full, n_walkers * n_steps)
    k = argmax(flat_lnLs)
    pbest_vec = flat_chains[k, :]
    pmed_vec = vecs.values[vi]
    unc = Dict{String, NamedTuple{(:plus, :minus), Tuple{Float64, Float64}}}()
    for i ∈ eachindex(pmed_vec)
        v, _unc = chain_uncertainty(flat_chains[:, i], [15.9, 50, 84.1] ./ 100)
        pmed_vec[i] = v
        unc[pnamesv[i]] = (;plus=_unc[1], minus=_unc[2])
    end
    
    for i ∈ eachindex(pbest_vec)
        pbest[pnamesv[i]].value = pbest_vec[i]
        pmed[pnamesv[i]].value = pmed_vec[i]
    end

    # lnLs
    lnLbest = flat_lnLs[k]
    lnLmed = obj_wrapper(pmed_vec)

    # Get result
    mcmc_result = (;pbest=pbest, lnLbest=lnLbest, pmed=pmed, lnLmed=lnLmed, n_steps=n_steps, τs=τmeds, chains=flat_chains, lnLs=flat_lnLs)

    return mcmc_result
end

function chain_uncertainty(flat_chain, p=[15.9, 50, 84.1])
    par_quantiles = quantile(flat_chain, p)
    v = par_quantiles[2]
    unc = diff(par_quantiles)
    return v, unc
end

function get_initial_walkers(post::RVPosterior, p0::Parameters)
    vecs = to_vecs(p0)
    vi = findall(vecs.vary)
    n_pars_vary = length(vi)
    search_scales = [get_scale(par) for par ∈ values(p0) if par.vary]
    n_walkers = 2 * n_pars_vary
    walkers = fill(NaN, (n_pars_vary, n_walkers))
    for i=1:n_walkers
        walkers[:, i] .= vecs.values[vi] .+ search_scales .* randn(n_pars_vary)
    end
    return walkers
end

include("autocorr.jl")