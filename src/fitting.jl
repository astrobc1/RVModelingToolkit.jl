using IterativeNelderMead

export run_mapfit, run_mcmc, model_comparison, planet_stats

using Infiltrator
using DSP
using Statistics
using AffineInvariantMCMC
using Statistics, NaNStatistics

export run_mapfit, run_mcmc

"""
    run_mapfit(post::RVPosterior, p0::Parameters)
Perform a maximum a posteriori fit using IterativeNelderMead.jl. Returns a NamedTuple with fields:
`pbest::Parameters` The best fit parameters.
`lnL::Float64` The nominal log-likelihood
`fcalls::Int` The number of function calls.
`simplex::Matrix{Float64}` The final simplex from the IterativeNelderMead solver.
`iteration::Int` The final iteration of the IterativeNelderMead solver.
"""
function run_mapfit(post::RVPosterior, p0::Parameters; output_path=nothing)
    println("Running MAP fit ...")
    p0_vecs = to_vecs(p0)
    ptest = deepcopy(p0)
    pbest = deepcopy(p0)
    scale_factors = [par.vary ? get_scale(par) : 0 for par in values(p0)]
    obj_wrapper = (x) -> begin
        set_values!(ptest, x)
        return -1 * compute_logaprob(post, ptest)
    end
    opt_result = IterativeNelderMead.optimize(obj_wrapper, p0_vecs.values;
                        vary=p0_vecs.vary,
                        scale_factors,
                        options=(;ftol_rel=1E-8)
                    )
    set_values!(pbest, opt_result.pbest)
    map_result = (;pbest=pbest, lnL=-1 * opt_result.fbest, fcalls=opt_result.fcalls, simplex=opt_result.simplex, iteration=opt_result.iterations)
    println("Completed MAP fit")
    if !isnothing(output_path)
        jldsave("$(output_path)map_result.jld"; map_result)
    end
    return map_result
end

"""
    run_mcmc(post::RVPosterior, p0::Parameters; n_burn_steps, τrel_thresh=0.01, n_min_steps=1000, n_τs_thresh=40, check_every=100, n_max_steps=100_000)
Performs an MCMC analysis with `n_burn_steps` followed by a dynamic number of steps, bounded by `n_min_steps` and `n_max_steps`. Convergence is determined through measuring the median auto-correlation time (# steps). Returns a NamedTuple with fields:
`pbest::Parameters` The set of parameters that yielded the best log-likelihood from all chains.
`lnLbest::Float64` The corresponding log-likelihood for the best tested parameters.
`pmed::Int` The parameters corresponding to the 50th percentile of each distribution.
`lnLmed::Matrix{Float64}` The corresponding log-likelihood for the median parameters.
`n_steps::Int` The number of MCMC steps performed (ignoring burn-in).
`τs::Vector{Float64}` The median auto-correlation times of each step.
`chains::Matrix{Float64}` The flattened MCMC chain of shape = `(n_steps, n_vary_parameters)`
`lnLs::Vector{Float64}` The corresponding log-likelihoods for the chain.
"""
function run_mcmc(post::RVPosterior, p0::Parameters; n_burn_steps, τrel_thresh=0.01, n_min_steps=1000, n_τs_thresh=40, check_every=100, n_max_steps=100_000, output_path=nothing)

    vecs = to_vecs(p0)
    vi = findall(vecs.vary)
    n_pars_vary = length(vi)
    pnamesv = collect(keys(p0))[vi]
    ptest = deepcopy(p0)
    pbest = deepcopy(p0)
    pmed = deepcopy(p0)

    obj_wrapper = (x::Vector{Float64}) -> begin
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
    #chain, lnLs = AffineInvariantMCMC.sample(obj_wrapper, n_walkers, walkers, n_burn_steps, 1)
    chains, lnLs = AffineInvariantMCMC.sample(obj_wrapper, n_walkers, walkers, n_burn_steps, 1, 2.0; filename="", load=false, save=false)
    
    println("Completed MCMC burn in")

    walkers = chains[:, :, end]
    
    println("Running MCMC ...")

    chains_full = fill(NaN, (0, n_walkers, n_pars_vary))
    lnLs_full = fill(NaN, (0, n_walkers))
    n_art_steps = Int(floor(2 * n_max_steps / check_every))

    τmeds = Float64[]
    τold = Inf
    line_old = ""
    ii = 0

    ti = time()

    for i=1:n_art_steps

        ii = (i - 1) * check_every + 1

        # Sample
        #chains, lnLs = AffineInvariantMCMC.sample(obj_wrapper, n_walkers, walkers, check_every, 1)
        chains, lnLs = AffineInvariantMCMC.sample(obj_wrapper, n_walkers, walkers, check_every, 1, 2.0; filename="", load=false, save=false)
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
        Δt = time() - ti
        line = " τ = $(round(τmed, digits=5)) / x $(n_τs_thresh), rel Δτ = $(round(τrel, digits=5)) / $(round(τrel_thresh, digits=5)), steps = $(n_min_steps) / $(ii) / $(n_max_steps), $(round(ii/Δt, digits=1)) steps / s"
        print("\r" * line * "     ")
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
    unc = Dict{String, NamedTuple{(:minus, :plus), Tuple{Float64, Float64}}}()
    for i ∈ eachindex(pmed_vec)
        r = chain_uncertainty(flat_chains[:, i], [15.9, 50, 84.1])
        pmed_vec[i] = r.value
        unc[pnamesv[i]] = (;minus=r.minus, plus=r.plus)
    end
    
    for i ∈ eachindex(pbest_vec)
        pbest[pnamesv[i]].value = pbest_vec[i]
        pmed[pnamesv[i]].value = pmed_vec[i]
    end

    # lnLs
    lnLbest = flat_lnLs[k]
    lnLmed = obj_wrapper(pmed_vec)

    # Get result
    mcmc_result = (;pbest=pbest, lnLbest=lnLbest, pmed=pmed, lnLmed=lnLmed, n_steps=n_steps, τs=τmeds, chains=flat_chains, lnLs=flat_lnLs, unc=unc)

    if !isnothing(output_path)
        jldsave("$(output_path)mcmc_result.jld"; mcmc_result)
    end

    return mcmc_result
end

function chain_uncertainty(flat_chain, p=[15.9, 50, 84.1])
    par_quantiles = quantile(flat_chain, p ./ 100)
    v = par_quantiles[2]
    unc = diff(par_quantiles)
    r = (;value=v, minus=unc[1], plus=unc[2])
    return r
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

function get_chain(chains, flat=false, thin=1, discard=0)
    v = [discard + thin - 1:self.iteration:thin]
    if flat
        s = list(v.shape[2:end])
        s[1] = prod(size(v)[1:2])
        return reshape(v, s)
    end
    return v
end

function get_autocorr_times(chain; discard=0, thin=1, c=5)
    return thin .* integrated_time((@view chain[1:thin:end, :, :]), c=c)
end

function integrated_time(chain; c=5)

    n_t, n_w, n_d = size(chain)
    τ_est = zeros(n_d)
    windows = zeros(Int, n_d)

    # Loop over parameters
    for d=1:n_d
        f = zeros(n_t)
        for k=1:n_w
            f .+= @views function_1d(chain[:, k, d])
        end
        f ./= n_w
        τs = 2 .* cumsum(f) .- 1
        windows[d] = auto_window(τs, c)
        τ_est[d] = τs[windows[d]]
    end

    return τ_est
end

function auto_window(τs, c)
    m = 1:length(τs) .< c .* τs
    if any(m)
        return argmin(m)
    end
    return length(τs)
end

function function_1d(x)
    n = next_pow_two(length(x))
    f = DSP.fft(x .- nanmean(x))
    acf = @views real.(DSP.ifft(f .* conj.(f))[1:length(x)])
    acf ./= acf[1]
    return acf
end

function next_pow_two(n)
    i = 1
    while i < n
        i = i << 1
    end
    return i
end