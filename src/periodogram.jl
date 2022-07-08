using ProgressMeter

export brute_force_periodogram

function brute_force_periodogram(post::RVPosterior, p0::Parameters, periods, planet_index; threads=true)
    @assert !p0["per$(planet_index)"].vary
    n = length(periods)
    p0cp = deepcopy(p0)
    results = Vector{NamedTuple}(undef, n)
    println("Running brute force periodogram for planet $(planet_index) ...")
    if threads
        prog = Progress(n)
        Threads.@threads for i=1:n
            results[i] = brute_force_periodogram_wrapper(post, p0cp, periods[i], planet_index)
            next!(prog)
        end
    else
        prog = Progress(n)
        for i=1:n
            results[i] = brute_force_periodogram_wrapper(post, p0cp, periods[i], planet_index)
            next!(prog)
        end
    end
    return results
end

function brute_force_periodogram_wrapper(post, pars, period, planet_index)
    pars["per$(planet_index)"].value = period
    map_result = run_mapfit(post, pars)
    pbest = map_result.pbest
    lnL = map_result.lnL
    n_free = num_varied(pbest)
    aicc = compute_aicc(post, pbest)
    bic = compute_bic(post, pbest)
    redχ2 = compute_redχ2(post, pbest)
    return (;pbest=pbest, lnL=lnL, redχ2=redχ2, aicc=aicc, bic=bic, n_free=n_free)
end